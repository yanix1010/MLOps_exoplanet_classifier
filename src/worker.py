import asyncio
import os
import time
from contextlib import asynccontextmanager

import asyncpg
import structlog
import uvicorn
from fastapi import FastAPI
from faststream import BaseMiddleware, ContextRepo, FastStream
from faststream.middlewares.acknowledgement.config import AckPolicy
from faststream.rabbit import Channel, RabbitBroker, RabbitMessage
from opentelemetry import trace
from opentelemetry.propagate import extract
from opentelemetry.trace import SpanKind
from structlog.contextvars import bound_contextvars

from src.astrobiology import AstrobiologyEngine
from src.config import (
    GRACEFUL_TIMEOUT_SECONDS,
    INFERENCE_DELAY_SECONDS,
    MAX_DELIVERY_ATTEMPTS,
    MODEL_ALIAS,
    MODEL_NAME,
    PREFETCH_COUNT,
    RABBITMQ_URL,
    RETRY_DELAY_SECONDS,
    WORKER_HEALTH_PORT,
)
from src.db import (
    close_db_pool,
    get_db_pool,
    init_db,
    save_task_failure,
    save_task_result,
)
from src.inference import ModelService, load_model_with_retries
from src.messaging import dlq_queue, tasks_queue
from src.telemetry import (
    configure_logging,
    configure_telemetry,
    inference_duration,
    operation,
    task_duration,
    worker_active,
    worker_attempts,
    worker_redeliveries,
)

logger = structlog.get_logger()

_model_service: ModelService | None = None
_health_server: uvicorn.Server | None = None

health_app = FastAPI(title="Worker Health Check API")


@health_app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "model_loaded": _model_service is not None,
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
    }


def compute_sync(
    service: ModelService,
    features: dict[str, float | None],
    delay_sec: float = 0.0,
) -> dict:
    if delay_sec > 0:
        time.sleep(delay_sec)

    if not service.knows_any_feature(features):
        raise ValueError("None of the provided features are known by the model")

    prediction_result = service.predict(features)
    radius = features.get("koi_prad")
    teq = features.get("koi_teq")
    esi_metrics = AstrobiologyEngine.compute_all_metrics(radius, teq)

    return {
        "prediction": prediction_result["prediction"],
        "probabilities": prediction_result["probabilities"],
        "missing_features": prediction_result["missing_features"],
        **esi_metrics,
    }


class TaskTelemetryMiddleware(BaseMiddleware):
    async def consume_scope(self, call_next, message):
        attempt = int(message.headers.get("x-delivery-count") or 0) + 1
        attributes = {
            "messaging.system": "rabbitmq",
            "messaging.destination.name": tasks_queue.name,
            "messaging.message.conversation_id": message.correlation_id or "",
            "messaging.delivery.attempt": attempt,
        }
        with (
            bound_contextvars(task_id=message.correlation_id),
            operation(
                "tasks process",
                context=extract(message.headers),
                kind=SpanKind.CONSUMER,
                attributes=attributes,
            ) as span,
        ):
            worker_active.add(1)
            if attempt > 1:
                worker_redeliveries.add(1)
            try:
                result = await call_next(message)
            except Exception as exc:  # noqa: BLE001 - FastStream still handles the nack
                outcome = "dlq" if attempt >= MAX_DELIVERY_ATTEMPTS else "error"
                worker_attempts.add(1, {"outcome": outcome})
                span.set_attribute("task.outcome", outcome)
                log_failure = logger.error if outcome == "dlq" else logger.warning
                log_failure(
                    "worker_delivery_limit_reached"
                    if outcome == "dlq"
                    else "worker_delivery_failed",
                    attempt=attempt,
                    exc_info=exc,
                )
                if outcome == "dlq":
                    span.add_event(
                        "delivery_limit_reached", {"messaging.dlq": dlq_queue.name}
                    )
                raise
            else:
                worker_attempts.add(1, {"outcome": "ok"})
                return result
            finally:
                worker_active.add(-1)


worker_channel = Channel(prefetch_count=PREFETCH_COUNT)
broker = RabbitBroker(
    RABBITMQ_URL,
    default_channel=worker_channel,
    graceful_timeout=GRACEFUL_TIMEOUT_SECONDS,
    middlewares=(TaskTelemetryMiddleware,),
    logger=None,
)


async def _handle_failure(task_id: str, attempt: int, exc: Exception) -> None:
    if attempt < MAX_DELIVERY_ATTEMPTS:
        trace.get_current_span().add_event(
            "retry_wait", {"retry.delay.seconds": RETRY_DELAY_SECONDS}
        )
        await asyncio.sleep(RETRY_DELAY_SECONDS)
        return

    try:
        with operation("db save failed task", kind=SpanKind.CLIENT):
            pool = await get_db_pool()
            await save_task_failure(pool, task_id, str(exc))
    except (OSError, RuntimeError, ValueError, asyncpg.PostgresError) as db_exc:
        logger.error("worker_db_failure_save_failed", exc_info=db_exc)


@broker.subscriber(
    tasks_queue, channel=worker_channel, ack_policy=AckPolicy.NACK_ON_ERROR
)
async def process_task(payload: dict, message: RabbitMessage) -> None:
    task_id = str(payload.get("task_id", "anonymous"))
    with bound_contextvars(task_id=task_id):
        attempt = int(message.headers.get("x-delivery-count") or 0) + 1
        started_at = time.time()

        enqueued_at = payload.get("enqueued_at")
        if isinstance(enqueued_at, (float, int)):
            trace.get_current_span().set_attribute(
                "task.queue_wait.seconds", max(0.0, started_at - enqueued_at)
            )
        logger.info("worker_task_received", attempt=attempt)

        try:
            features = payload.get("features")
            if not isinstance(features, dict):
                raise ValueError("Message has no features object")

            inference_started = time.perf_counter()
            try:
                with operation(
                    "inference",
                    attributes={
                        "model.alias": MODEL_ALIAS,
                        "model.version": _model_service.model_version,
                    },
                ):
                    # asyncio.to_thread copies the active context into the executor.
                    result = await asyncio.to_thread(
                        compute_sync, _model_service, features, INFERENCE_DELAY_SECONDS
                    )
            finally:
                inference_duration.record(time.perf_counter() - inference_started)

            with operation("db save result", kind=SpanKind.CLIENT):
                pool = await get_db_pool()
                await save_task_result(
                    pool,
                    task_id,
                    result["prediction"],
                    result["probabilities"],
                    esi=result.get("esi"),
                    esi_percent=result.get("esi_percent"),
                    is_habitable_zone=result.get("is_habitable_zone"),
                    is_rocky=result.get("is_rocky"),
                    planet_category=result.get("planet_category"),
                )
        except Exception as exc:
            await _handle_failure(task_id, attempt, exc)
            raise

        finished_at = time.time()
        total_sec = None
        if isinstance(enqueued_at, (float, int)):
            total_sec = max(0.0, finished_at - enqueued_at)
            task_duration.record(total_sec)
        logger.info(
            "worker_task_completed",
            prediction=result.get("prediction"),
            esi_percent=result.get("esi_percent"),
            process_sec=round(finished_at - started_at, 3),
            total_sec=total_sec,
        )


@asynccontextmanager
async def lifespan(context: ContextRepo):
    global _model_service, _health_server
    logger.info(
        "worker_starting",
        model_name=MODEL_NAME,
        model_alias=MODEL_ALIAS,
        prefetch_count=PREFETCH_COUNT,
        inference_delay_seconds=INFERENCE_DELAY_SECONDS,
    )

    _model_service = await load_model_with_retries()
    if _model_service is None:
        raise RuntimeError(f"Model {MODEL_NAME}@{MODEL_ALIAS} is not loaded")

    logger.info(
        "worker_model_loaded",
        model_name=_model_service.model_name,
        version=_model_service.model_version,
        alias=MODEL_ALIAS,
    )
    context.set_global("model_service", _model_service)
    await init_db()

    server_task = None
    if os.getenv("ENABLE_WORKER_HEALTH_SERVER", "true").lower() in ("true", "1"):
        try:
            cfg = uvicorn.Config(
                health_app,
                host="0.0.0.0",
                port=WORKER_HEALTH_PORT,
                log_config=None,
                access_log=False,
            )
            _health_server = uvicorn.Server(cfg)
            server_task = asyncio.create_task(_health_server.serve())
            logger.info("worker_health_server_started", port=WORKER_HEALTH_PORT)
        except (OSError, RuntimeError) as exc:
            logger.warning("worker_health_server_start_failed", exc_info=exc)

    try:
        yield
    finally:
        if _health_server:
            _health_server.should_exit = True
        if server_task:
            try:
                await server_task
            except (asyncio.CancelledError, OSError, RuntimeError):
                pass
        await close_db_pool()
        logger.info("worker_shutdown_complete")


app = FastStream(broker, lifespan=lifespan)


@app.on_shutdown
async def log_shutdown_started() -> None:
    logger.info("worker_shutting_down")


@app.after_startup
async def declare_dlq() -> None:
    await broker.declare_queue(dlq_queue)


if __name__ == "__main__":
    configure_logging("worker")
    configure_telemetry("worker")
    asyncio.run(app.run())
