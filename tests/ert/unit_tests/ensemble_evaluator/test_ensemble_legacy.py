import asyncio
import os
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from _ert.events import EESnapshot, EESnapshotUpdate, EETerminated
from ert.ensemble_evaluator import EnsembleEvaluator, Monitor, identifiers, state
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.scheduler import Scheduler


@pytest.fixture
def evaluator_to_use():
    @asynccontextmanager
    async def run_evaluator(ensemble, ee_config):
        evaluator = EnsembleEvaluator(ensemble, ee_config)
        run_task = asyncio.create_task(evaluator.run_and_get_successful_realizations())
        await evaluator._server_started
        try:
            yield evaluator
        finally:
            evaluator.stop()
            await run_task

    return run_evaluator


@pytest.mark.integration_test
@pytest.mark.timeout(60)
@pytest.mark.asyncio
async def test_run_legacy_ensemble(
    tmpdir, make_ensemble, monkeypatch, evaluator_to_use
):
    num_reals = 2
    with tmpdir.as_cwd():
        ensemble = make_ensemble(monkeypatch, tmpdir, num_reals, 2)
        config = EvaluatorServerConfig(use_token=False)
        async with (
            evaluator_to_use(ensemble, config) as evaluator,
            Monitor(config.get_uri(), config.token) as monitor,
        ):
            async for event in monitor.track():
                if type(event) in {
                    EESnapshotUpdate,
                    EESnapshot,
                } and event.snapshot.get(identifiers.STATUS) in {
                    state.ENSEMBLE_STATE_FAILED,
                    state.ENSEMBLE_STATE_STOPPED,
                }:
                    await monitor.signal_done()
            assert evaluator._ensemble.status == state.ENSEMBLE_STATE_STOPPED
            assert len(evaluator._ensemble.get_successful_realizations()) == num_reals

        # realisations should finish, each creating a status-file
        for i in range(num_reals):
            assert os.path.isfile(f"real_{i}/status.txt")


@pytest.mark.integration_test
@pytest.mark.timeout(60)
async def test_run_and_cancel_legacy_ensemble(
    tmpdir, make_ensemble, monkeypatch, evaluator_to_use
):
    num_reals = 2
    with tmpdir.as_cwd():
        ensemble = make_ensemble(monkeypatch, tmpdir, num_reals, 2, job_sleep=40)
        config = EvaluatorServerConfig(use_token=False)

        terminated_event = False

        async with (
            evaluator_to_use(ensemble, config) as evaluator,
            Monitor(config.get_uri(), config.token) as monitor,
        ):
            # on lesser hardware the realizations might be killed by max_runtime
            # and the ensemble is set to STOPPED
            monitor._receiver_timeout = 10.0
            cancel = True
            await evaluator._ensemble._scheduler._running.wait()
            async for event in monitor.track(heartbeat_interval=0.1):
                # Cancel the ensemble upon the arrival of the first event
                if cancel:
                    await monitor.signal_cancel()
                    cancel = False
                if type(event) is EETerminated:
                    terminated_event = True

        if terminated_event:
            assert evaluator._ensemble.status == state.ENSEMBLE_STATE_CANCELLED
        else:
            assert evaluator._ensemble.status == state.ENSEMBLE_STATE_STOPPED

        # realisations should not finish, thus not creating a status-file
        for i in range(num_reals):
            assert not os.path.isfile(f"real_{i}/status.txt")


async def test_queue_config_properties_propagated_to_scheduler(
    tmpdir, make_ensemble, monkeypatch
):
    num_reals = 1
    mocked_scheduler = MagicMock()
    mocked_scheduler.__class__ = Scheduler
    monkeypatch.setattr(Scheduler, "__init__", mocked_scheduler)
    ensemble = make_ensemble(monkeypatch, tmpdir, num_reals, 2)
    ensemble._config = MagicMock()
    ensemble._scheduler = mocked_scheduler

    # The properties we want to propagate from QueueConfig to the Scheduler object:
    ensemble._queue_config.max_submit = 55
    ensemble._queue_config.queue_options.submit_sleep = 33
    ensemble._queue_config.queue_options.max_running = 44

    # The function under test:
    await ensemble.evaluate(config=MagicMock())

    # Assert properties successfully propagated:
    assert Scheduler.__init__.call_args.kwargs["submit_sleep"] == 33
    assert Scheduler.__init__.call_args.kwargs["max_running"] == 44
    assert Scheduler.__init__.call_args.kwargs["max_submit"] == 55
