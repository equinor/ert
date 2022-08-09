from typing import AsyncContextManager, Awaitable, Callable
from unittest.mock import AsyncMock

import pytest
from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent
from websockets.client import WebSocketClientProtocol

import ert.experiment_server
from ert.experiment_server._experiment_protocol import Experiment

# All test coroutines will be treated as marked.
pytestmark = pytest.mark.asyncio


async def test_receiving_event_from_cluster(
    experiment_server_ctx: AsyncContextManager[ert.experiment_server.ExperimentServer],
    dispatcher_factory: AsyncContextManager[
        Callable[
            [ert.experiment_server.ExperimentServer], Awaitable[WebSocketClientProtocol]
        ]
    ],
):
    async with experiment_server_ctx as experiment_server:
        experiment = AsyncMock(Experiment)
        experiment_server.add_experiment(experiment)

        async with dispatcher_factory as make_dispatcher:
            dispatcher = await make_dispatcher(experiment_server)

            event = CloudEvent(
                {
                    "type": "test.event",
                    "source": "test_receiving_event_from_cluster",
                }
            )
            await dispatcher.send(to_json(event).decode())

    experiment.dispatch.assert_awaited_once_with(event, 0)


async def test_successful_run(
    experiment_server_ctx: AsyncContextManager[ert.experiment_server.ExperimentServer],
    capsys,
):
    async with experiment_server_ctx as experiment_server:
        experiment = AsyncMock(Experiment)
        experiment.successful_realizations.return_value = 5
        id_ = experiment_server.add_experiment(experiment)
        await experiment_server.run_experiment(id_)

    captured = capsys.readouterr()
    assert captured.out == "Successful realizations: 5\n"


async def test_failed_run(
    experiment_server_ctx: AsyncContextManager[ert.experiment_server.ExperimentServer],
    capsys,
):
    async with experiment_server_ctx as experiment_server:
        experiment = AsyncMock(Experiment)
        experiment.run.side_effect = RuntimeError("boom")
        id_ = experiment_server.add_experiment(experiment)

        with pytest.raises(RuntimeError, match="boom"):
            await experiment_server.run_experiment(id_)

    captured = capsys.readouterr()
    assert captured.out == "Experiment failed: boom\n"
