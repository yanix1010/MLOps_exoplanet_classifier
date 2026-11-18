import os
import time
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request
from opentelemetry.propagate import extract
from opentelemetry.trace import SpanKind, StatusCode

from src.api.v1.health import router as health_v1_router
from src.api.v1.tasks import router as tasks_v1_router
from src.config import QUEUE_MAX_LENGTH, get_app_version
from src.db import close_db_pool, init_db
from src.telemetry import (
    configure_logging,
    configure_telemetry,
    http_duration,
    http_requests,
    operation,
    queue_limit,
)

logger = structlog.get_logger()
API_V1_PREFIX = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging("app")
    configure_telemetry("app")
    queue_limit.set(max(0, QUEUE_MAX_LENGTH - 1))
    await init_db()
    try:
        yield
    finally:
        broker = getattr(app.state, "rabbit_broker", None)
        if broker is not None:
            try:
                await broker.stop()
            except (OSError, RuntimeError) as exc:
                logger.warning("broker_close_warning", exc_info=exc)
        await close_db_pool()


def create_app() -> FastAPI:
    app = FastAPI(
        title="MLOps Project API",
        version=get_app_version(),
        description="FastAPI сервис с мониторингом состояния и компонентов",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def observe_request(request: Request, call_next):
        if request.url.path in ("/healthz", "/api/v1/health"):
            return await call_next(request)

        started = time.perf_counter()
        code = 500
        method = request.method
        if method not in {"GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"}:
            method = "OTHER"
        with operation(
            method, context=extract(request.headers), kind=SpanKind.SERVER
        ) as span:
            try:
                response = await call_next(request)
                code = response.status_code
                return response
            finally:
                route = getattr(request.scope.get("route"), "path", "unmatched")
                if (
                    request.url.path.startswith(f"{API_V1_PREFIX}/")
                    and route.startswith("/")
                    and not route.startswith(f"{API_V1_PREFIX}/")
                ):
                    route = f"{API_V1_PREFIX}{route}"
                span.update_name(f"{method} {route}")
                span.set_attributes(
                    {
                        "http.request.method": method,
                        "http.route": route,
                        "http.response.status_code": code,
                    }
                )
                if code >= 500:
                    span.set_status(StatusCode.ERROR)
                labels = {"route": route, "status": str(code), "method": method}
                http_requests.add(1, labels)
                http_duration.record(time.perf_counter() - started, labels)

    @app.get("/healthz", tags=["Probes"])
    async def healthz() -> dict[str, str]:
        logger.debug("liveness_probe_called", endpoint="/healthz", status="ok")
        return {"status": "ok"}

    app.include_router(health_v1_router, prefix=API_V1_PREFIX)
    app.include_router(tasks_v1_router, prefix=API_V1_PREFIX)

    return app


app = create_app()


if __name__ == "__main__":
    configure_logging("app")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_config=None,
        access_log=False,
    )
