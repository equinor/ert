import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from _ert.events import EESnapshot, EESnapshotUpdate, Event
from ert.config import QueueConfig
from ert.ensemble_evaluator import EnsembleEvaluator, identifiers, state
from ert.ensemble_evaluator._ensemble import LegacyEnsemble
from ert.ensemble_evaluator.config import EvaluatorServerConfig
from ert.scheduler import Scheduler


@asynccontextmanager
async def run_evaluator(
    ensemble: LegacyEnsemble, ee_config: EvaluatorServerConfig
) -> AsyncGenerator[tuple[EnsembleEvaluator, asyncio.Queue[Event]], None]:
    event_queue = asyncio.Queue()
    evaluator = EnsembleEvaluator(
        ensemble, ee_config, event_handler=event_queue.put_nowait
    )
    run_task = asyncio.create_task(evaluator.run_and_get_successful_realizations())
    await evaluator._server_started
    try:
        yield (evaluator, event_queue)
    finally:
        evaluator.stop()
        await run_task


@pytest.mark.integration_test
@pytest.mark.timeout(60)
@pytest.mark.asyncio
async def test_run_legacy_ensemble(tmpdir, make_ensemble, monkeypatch):
    num_reals = 2
    with tmpdir.as_cwd():
        ensemble = make_ensemble(monkeypatch, tmpdir, num_reals, 2)
        config = EvaluatorServerConfig(use_token=False)
        async with run_evaluator(ensemble, config) as (evaluator, event_queue):
            while True:
                event = await event_queue.get()
                if type(event) in {
                    EESnapshotUpdate,
                    EESnapshot,
                } and event.snapshot.get(identifiers.STATUS) in {
                    state.ENSEMBLE_STATE_FAILED,
                    state.ENSEMBLE_STATE_STOPPED,
                }:
                    break
            assert evaluator._ensemble.status == state.ENSEMBLE_STATE_STOPPED
            assert len(evaluator._ensemble.get_successful_realizations()) == num_reals

        # realisations should finish, each creating a status-file
        for i in range(num_reals):
            assert os.path.isfile(f"real_{i}/status.txt")


@pytest.mark.integration_test
@pytest.mark.timeout(60)
async def test_run_and_cancel_legacy_ensemble(tmpdir, make_ensemble, monkeypatch):
    num_reals = 2
    with tmpdir.as_cwd():
        ensemble = make_ensemble(monkeypatch, tmpdir, num_reals, 2, job_sleep=40)
        config = EvaluatorServerConfig(use_token=False)

        async with (
            run_evaluator(ensemble, config) as (evaluator, event_queue),
        ):
            # Wait for ensemble to start evaluating
            while evaluator._ensemble._scheduler is None:  # noqa: ASYNC110
                await asyncio.sleep(0.1)

            await evaluator._ensemble._scheduler._running.wait()

            # Cancel the ensemble upon the arrival of the first event
            await event_queue.get()
            await evaluator.cancel_gracefully()
            await evaluator._is_done.wait()
            if await evaluator._monitoring_result == False:
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
    monkeypatch.setattr(QueueConfig, "submit_sleep", 33)
    monkeypatch.setattr(QueueConfig, "max_running", 44)
    ensemble._queue_config.max_submit = 55

    async def dummy_unary_send(_):
        return

    # The function under test:
    await ensemble._evaluate_inner(dummy_unary_send)

    # Assert properties successfully propagated:
    assert Scheduler.__init__.call_args.kwargs["submit_sleep"] == 33
    assert Scheduler.__init__.call_args.kwargs["max_running"] == 44
    assert Scheduler.__init__.call_args.kwargs["max_submit"] == 55
