import asyncio
import os
import uuid
from unittest.mock import patch

import pytest
import websockets
from cloudevents.conversion import to_json
from cloudevents.http import CloudEvent
from websockets.exceptions import ConnectionClosed

from ert.ensemble_evaluator import identifiers, state
from ert.ensemble_evaluator._wait_for_evaluator import wait_for_evaluator
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.ensemble_evaluator.evaluator import EnsembleEvaluator


@pytest.mark.timeout(60)
@pytest.mark.asyncio
async def test_run_legacy_ensemble(tmpdir, make_ensemble_builder):
    num_reals = 2
    custom_port_range = range(1024, 65535)
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(tmpdir, num_reals, 2).build()
        config = EvaluatorServerConfig(
            custom_port_range=custom_port_range,
            custom_host="127.0.0.1",
            use_token=False,
            generate_cert=False,
        )
        server = EnsembleEvaluator(ensemble, config, 0)
        server_task = asyncio.create_task(server.evaluator_server())

        await ensemble.evaluate_async(config, experiment_id=None)
        await server.stop()
        await server_task

        assert ensemble.status == state.ENSEMBLE_STATE_STOPPED
        assert ensemble.get_successful_realizations() == num_reals

        # realisations should finish, each creating a status-file
        for i in range(num_reals):
            assert os.path.isfile(f"real_{i}/status.txt")


@pytest.mark.timeout(60)
@pytest.mark.asyncio
async def test_run_and_cancel_legacy_ensemble(tmpdir, make_ensemble_builder):
    num_reals = 2
    custom_port_range = range(1024, 65535)
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(tmpdir, num_reals, 2, job_sleep=30).build()
        config = EvaluatorServerConfig(
            custom_port_range=custom_port_range,
            custom_host="127.0.0.1",
            use_token=False,
            generate_cert=False,
        )

        server = EnsembleEvaluator(ensemble, config, 0)

        server_task = asyncio.create_task(server.evaluator_server())
        con_info = config.get_connection_info()
        await wait_for_evaluator(
            base_url=con_info.url, token=con_info.token, cert=con_info.cert
        )
        evaluation_task = asyncio.create_task(
            ensemble.evaluate_async(config, experiment_id=None)
        )
        cancel_event = CloudEvent(
            {
                "type": identifiers.EVTYPE_EE_USER_CANCEL,
                "source": f"/ert/monitor/0",
                "id": str(uuid.uuid1()),
            }
        )

        async with websockets.connect(f"{config.client_uri}") as websocket:
            await websocket.send(to_json(cancel_event))

        await evaluation_task
        await server.stop()
        await server_task

        assert ensemble.status == state.ENSEMBLE_STATE_CANCELLED

        # realisations should not finish, thus not creating a status-file
        for i in range(num_reals):
            assert not os.path.isfile(f"real_{i}/status.txt")


@pytest.mark.timeout(60)
@pytest.mark.asyncio
async def test_run_legacy_ensemble_exception(tmpdir, make_ensemble_builder):
    num_reals = 2
    custom_port_range = range(1024, 65535)
    with tmpdir.as_cwd():
        ensemble = make_ensemble_builder(tmpdir, num_reals, 2).build()
        config = EvaluatorServerConfig(
            custom_port_range=custom_port_range,
            custom_host="127.0.0.1",
            use_token=False,
            generate_cert=False,
        )
        evaluator = EnsembleEvaluator(ensemble, config, 0)
        server = EnsembleEvaluator(ensemble, config, 0)
        server_task = asyncio.create_task(server.evaluator_server())

        con_info = config.get_connection_info()
        await wait_for_evaluator(
            base_url=con_info.url, token=con_info.token, cert=con_info.cert
        )

        with patch.object(ensemble._job_queue, "submit_complete") as faulty_queue:
            faulty_queue.side_effect = RuntimeError()
            await ensemble.evaluate_async(config, experiment_id=None)

        await server.stop()
        await server_task

        assert evaluator._ensemble.status == state.ENSEMBLE_STATE_FAILED

        # realisations should not finish, thus not creating a status-file
        for i in range(num_reals):
            assert not os.path.isfile(f"real_{i}/status.txt")
