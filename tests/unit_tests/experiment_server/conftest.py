from contextlib import asynccontextmanager
from typing import AsyncGenerator

import pytest

import ert.experiment_server
from ert.ensemble_evaluator.config import EvaluatorServerConfig


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
