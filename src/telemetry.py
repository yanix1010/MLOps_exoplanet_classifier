import logging
import os
import re
import sys
import traceback
from contextlib import contextmanager
from functools import lru_cache

import structlog
from opentelemetry import metrics, trace
from opentelemetry.metrics import Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import StatusCode

from src.config import INFERENCE_DELAY_SECONDS, get_app_version

tracer = trace.get_tracer("exoplanet")
meter = metrics.get_meter("exoplanet")
logger = structlog.get_logger()

http_requests = meter.create_counter("exoplanet_api_requests")
http_duration = meter.create_histogram(
    "exoplanet_api_duration",
    unit="s",
    explicit_bucket_boundaries_advisory=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1,
        2,
        5,
    ),
)
published = meter.create_counter("exoplanet_producer_published")
rejections = meter.create_counter("exoplanet_producer_rejections")
producer_wait = meter.create_counter("exoplanet_producer_wait", unit="s")
worker_attempts = meter.create_counter("exoplanet_worker_attempts")
worker_redeliveries = meter.create_counter("exoplanet_worker_redeliveries")
worker_active = meter.create_up_down_counter("exoplanet_worker_active")
duration_base = max(0.01, INFERENCE_DELAY_SECONDS)
inference_duration = meter.create_histogram(
    "exoplanet_inference_duration",
    unit="s",
    explicit_bucket_boundaries_advisory=tuple(
        duration_base * ratio
        for ratio in (0.1, 0.25, 0.5, 0.9, 1, 1.1, 1.25, 1.5, 2, 4, 10)
    ),
)
task_duration = meter.create_histogram(
    "exoplanet_task_duration",
    unit="s",
    explicit_bucket_boundaries_advisory=(
        0.1,
        0.5,
        1,
        2,
        5,
        10,
        15,
        20,
        30,
        60,
        120,
        300,
    ),
)
queue_limit = meter.create_gauge("exoplanet_queue_limit")


def _log_context(_, __, event):
    context = trace.get_current_span().get_span_context()
    if context.is_valid:
        event["trace_id"] = format(context.trace_id, "032x")
        event["span_id"] = format(context.span_id, "016x")
    return event


def _safe_exception(_, __, event):
    exc_info = event.pop("exc_info", None)
    if exc_info is True:
        exc_info = sys.exc_info()
    elif isinstance(exc_info, BaseException):
        exc_info = (type(exc_info), exc_info, exc_info.__traceback__)
    if exc_info and exc_info[0]:
        event["exception"] = {
            "type": exc_info[0].__name__,
            "frames": [
                {"file": frame.filename, "line": frame.lineno, "function": frame.name}
                for frame in traceback.extract_tb(exc_info[2])
            ],
        }
    return event


def _redact_urls(_, __, event):
    for key, value in event.items():
        if isinstance(value, str):
            event[key] = re.sub(
                r"([a-zA-Z][\w+.-]*://)[^\s/@]+@", r"\1[redacted]@", value
            )
    return event


def configure_logging(service: str) -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    shared = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _log_context,
        _safe_exception,
    ]
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                _redact_urls,
                lambda _, __, event: {"service": service, **event},
                structlog.processors.JSONRenderer(),
            ],
        )
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "faststream", "mlflow"):
        library = logging.getLogger(name)
        library.handlers.clear()
        library.setLevel(logging.NOTSET)
        library.propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _process_memory(_):
    if sys.platform == "linux":
        # VmRSS includes the model and native allocations from NumPy/scikit-learn.
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return [Observation(int(line.split()[1]) * 1024)]
    return []


@lru_cache(maxsize=1)
def configure_telemetry(service: str) -> None:
    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return

    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    # Resource.create applies OTEL_SERVICE_NAME and OTEL_RESOURCE_ATTRIBUTES.
    resource = Resource.create(
        {
            "service.name": service,
            "service.version": get_app_version(),
            "deployment.environment.name": "local",
            "service.instance.id": os.getenv("HOSTNAME") or str(os.getpid()),
        }
    )
    if os.getenv("OTEL_TRACES_EXPORTER") != "none":
        span_provider = TracerProvider(resource=resource)
        span_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(timeout=2))
        )
        trace.set_tracer_provider(span_provider)

    if os.getenv("OTEL_METRICS_EXPORTER") != "none":
        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(timeout=2),
            export_interval_millis=5000,
            export_timeout_millis=2000,
        )
        metrics.set_meter_provider(
            MeterProvider(resource=resource, metric_readers=[reader])
        )
        meter.create_observable_gauge(
            "exoplanet_process_memory", callbacks=[_process_memory], unit="By"
        )


def record_error(span, exc: Exception) -> None:
    # Preserve traceback locations without exporting payloads or connection strings.
    safe = Exception(type(exc).__name__).with_traceback(exc.__traceback__)
    span.record_exception(safe, {"error.type": type(exc).__name__})
    span.set_status(StatusCode.ERROR)


@contextmanager
def operation(name, **options):
    with tracer.start_as_current_span(
        name, record_exception=False, set_status_on_exception=False, **options
    ) as span:
        try:
            yield span
        except Exception as exc:  # noqa: BLE001 - record and re-raise the original error
            record_error(span, exc)
            raise
