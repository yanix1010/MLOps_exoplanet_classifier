from unittest.mock import AsyncMock, patch

import pytest
from faststream.rabbit import TestRabbitBroker
from httpx import ASGITransport, AsyncClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from src import worker
from src.app import app
from tests.test_worker import make_dummy_service


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with patch("src.telemetry.tracer", provider.get_tracer("test")):
        yield exporter
    provider.shutdown()


@pytest.mark.asyncio
async def test_task_trace_crosses_the_rabbitmq_message(spans):
    async with (
        TestRabbitBroker(worker.broker),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        with (
            patch("src.worker._model_service", make_dummy_service()),
            patch("src.worker.INFERENCE_DELAY_SECONDS", 0),
            patch("src.worker.get_db_pool", new=AsyncMock()),
            patch("src.worker.save_task_result", new=AsyncMock()),
            patch("src.api.v1.tasks.get_or_create_broker", return_value=worker.broker),
            patch("src.api.v1.tasks.get_db_pool", new=AsyncMock()),
            patch("src.api.v1.tasks.create_pending_task", new=AsyncMock()),
        ):
            response = await client.post(
                "/api/v1/tasks", json={"features": {"koi_period": 1.0}}
            )

    assert response.status_code == 202
    by_name = {span.name: span for span in spans.get_finished_spans()}
    expected = ("POST /api/v1/tasks", "tasks publish", "tasks process")
    assert len({by_name[name].context.trace_id for name in expected}) == 1
    assert (
        by_name["tasks process"].parent.span_id
        == by_name["tasks publish"].context.span_id
    )
    assert (
        by_name["tasks publish"].parent.span_id == by_name[expected[0]].context.span_id
    )
    assert (
        by_name["tasks process"].attributes["messaging.message.conversation_id"]
        == (response.json()["task_id"])
    )
    for name in ("inference", "db save result"):
        assert by_name[name].parent.span_id == by_name["tasks process"].context.span_id
    assert by_name["inference"].attributes["model.version"] == "1"


@pytest.mark.asyncio
async def test_broker_unavailable_is_visible_as_a_failed_http_span(spans):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        with patch(
            "src.api.v1.tasks.get_or_create_broker", side_effect=OSError("offline")
        ):
            response = await client.post(
                "/api/v1/tasks", json={"features": {"koi_period": 1.0}}
            )

    assert response.status_code == 503
    (span,) = spans.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["http.response.status_code"] == 503
