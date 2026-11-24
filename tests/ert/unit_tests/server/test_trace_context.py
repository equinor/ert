import asyncio

import httpx
from opentelemetry import trace

from ert.server.app import app
from ert.trace import tracer


@app.get("/_test/current_trace_id")
async def _current_trace_id() -> str:
    return trace.format_trace_id(trace.get_current_span().get_span_context().trace_id)


def test_that_requests_are_handled_in_the_trace_context_of_the_server():
    async def request_trace_id() -> str:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return (await client.get("/_test/current_trace_id")).json()

    with tracer.start_as_current_span("run_storage_server") as server_span:
        expected = trace.format_trace_id(server_span.get_span_context().trace_id)
        assert asyncio.run(request_trace_id()) == expected
