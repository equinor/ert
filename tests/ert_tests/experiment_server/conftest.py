import asyncio
import sys
from typing import AsyncGenerator, Awaitable, Callable, Optional
import pytest
from websockets.client import WebSocketClientProtocol, connect
import ert.experiment_server
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig

if sys.version_info < (3, 7):
    from async_generator import asynccontextmanager
else:
    from contextlib import asynccontextmanager


@pytest.fixture
@asynccontextmanager
async def experiment_server_ctx() -> AsyncGenerator[
    ert.experiment_server.ExperimentServer, None
]:
    config = EvaluatorServerConfig(
        custom_port_range=range(1024, 65535),
        custom_host="127.0.0.1",
        use_token=False,
        generate_cert=False,
    )
    server = ert.experiment_server.ExperimentServer(config)
    yield server
    await server.stop()


@pytest.fixture
@asynccontextmanager
async def dispatcher_factory() -> AsyncGenerator[
    Callable[
        [ert.experiment_server.ExperimentServer], Awaitable[WebSocketClientProtocol]
    ],
    None,
]:
    connection: Optional[WebSocketClientProtocol] = None

    async def _make_dispatcher(
        server: ert.experiment_server.ExperimentServer,
    ) -> WebSocketClientProtocol:
        nonlocal connection

        async def _wait_for_connection() -> WebSocketClientProtocol:
            async for websocket in connect(server._config.dispatch_uri):
                return websocket
            raise RuntimeError

        connection = await asyncio.wait_for(_wait_for_connection(), timeout=60)
        return connection

    yield _make_dispatcher

    if connection:
        await connection.close()
