import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from _ert.events import EEEvent
from ert.config import QueueConfig
from ert.ensemble_evaluator import EnsembleEvaluator, state
from ert.ensemble_evaluator.evaluator import UserCancelled
from ert.scheduler import Scheduler, job
from ert.scheduler.job import Job


@pytest.mark.integration_test
@pytest.mark.timeout(60)
@pytest.mark.asyncio
async def test_run_legacy_ensemble(
    tmpdir, make_ensemble, monkeypatch, evaluator_to_use
):
    num_reals = 2
    # Skip waiting for stdout/err in job
    mocked_stdouterr_parser = AsyncMock(
        return_value=Job.DEFAULT_FILE_VERIFICATION_TIMEOUT
    )
    monkeypatch.setattr(job, "log_warnings_from_forward_model", mocked_stdouterr_parser)
    with tmpdir.as_cwd():
        ensemble = make_ensemble(monkeypatch, tmpdir, num_reals, 2)

        async with (
            evaluator_to_use(ensemble=ensemble) as evaluator,
        ):
            assert isinstance(evaluator, EnsembleEvaluator)
            assert await evaluator.wait_for_evaluation_result()
            assert evaluator._ensemble.status == state.ENSEMBLE_STATE_STOPPED
            assert len(evaluator._ensemble.get_successful_realizations()) == num_reals

        # realisations should finish, each creating a status-file
        for i in range(num_reals):
            assert os.path.isfile(f"real_{i}/status.txt")


@pytest.mark.integration_test
@pytest.mark.timeout(30)
@pytest.mark.usefixtures("use_tmpdir")
async def test_run_and_cancel_legacy_ensemble(
    make_ensemble, monkeypatch, evaluator_to_use, tmpdir
):
    num_reals = 2
    event_queue: asyncio.Queue[EEEvent] = asyncio.Queue()
    ensemble = make_ensemble(monkeypatch, tmpdir, num_reals, 2, job_sleep=40)
    async with evaluator_to_use(
        event_handler=event_queue.put_nowait, ensemble=ensemble
    ) as evaluator:
        assert isinstance(evaluator, EnsembleEvaluator)
        evaluator._publisher_receiving_timeout = 10.0

        _ = await event_queue.get()
        # Cancel the ensemble upon the arrival of the first event
        evaluator._end_queue.put("END")
        try:
            await evaluator.wait_for_evaluation_result()
            assert evaluator._ensemble.status == state.ENSEMBLE_STATE_STOPPED
        except UserCancelled:
            assert evaluator._ensemble.status == state.ENSEMBLE_STATE_CANCELLED
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

    # The function under test:
    await ensemble.evaluate(config=MagicMock())

    # Assert properties successfully propagated:
    assert Scheduler.__init__.call_args.kwargs["submit_sleep"] == 33
    assert Scheduler.__init__.call_args.kwargs["max_running"] == 44
    assert Scheduler.__init__.call_args.kwargs["max_submit"] == 55
