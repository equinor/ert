import asyncio
import itertools
import json
import logging
import random
import shutil
import time
from functools import partial
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from _ert.events import (
    RealizationFailed,
    RealizationStoppedLongRunning,
    RealizationTimeout,
)
from ert.config import QueueConfig, QueueSystem
from ert.ensemble_evaluator import Realization
from ert.run_arg import RunArg
from ert.scheduler import LsfDriver, OpenPBSDriver, create_driver, job, scheduler
from ert.scheduler.job import Job, JobState
from ert.storage.load_status import LoadResult


def create_jobs_json(realization: Realization) -> None:
    jobs = {
        "global_environment": {},
        "config_path": "/dev/null",
        "config_file": "/dev/null",
        "jobList": [
            {
                "name": fm_step.name,
                "executable": fm_step.executable,
                "argList": fm_step.arglist,
            }
            for fm_step in realization.fm_steps
        ],
        "run_id": "0",
        "ert_pid": "0",
        "real_id": str(realization.iens),
    }
    realization_run_path = Path(realization.run_arg.runpath)
    realization_run_path.mkdir()
    with open(realization_run_path / "jobs.json", mode="w", encoding="utf-8") as f:
        json.dump(jobs, f)


@pytest.fixture
def realization(storage, tmp_path):
    ensemble = storage.create_experiment().create_ensemble(name="foo", ensemble_size=1)
    return create_stub_realization(ensemble, tmp_path, 0)


def create_stub_realization(ensemble, base_path: Path, iens) -> Realization:
    run_arg = RunArg(
        run_id="",
        ensemble_storage=ensemble,
        iens=iens,
        itr=0,
        runpath=str(base_path / f"realization-{iens}"),
        job_name="",
    )

    realization = Realization(
        iens=iens,
        fm_steps=[],
        active=True,
        max_runtime=None,
        run_arg=run_arg,
        num_cpu=1,
        job_script=str(shutil.which("fm_dispatch.py")),
        realization_memory=0,
    )
    return realization


async def test_empty(mock_driver):
    sch = scheduler.Scheduler(mock_driver())
    assert await sch.execute()


async def test_single_job(realization, mock_driver):
    future = asyncio.Future()

    async def init(iens, *args, **kwargs):
        future.set_result(iens)

    driver = mock_driver(init=init)

    sch = scheduler.Scheduler(driver, [realization])

    assert await sch.execute()
    assert await future == realization.iens


async def test_cancel(realization, mock_driver):
    pre = asyncio.Event()
    post = asyncio.Event()
    killed = False

    async def wait():
        pre.set()
        await asyncio.sleep(10)
        post.set()

    async def kill():
        nonlocal killed
        killed = True

    driver = mock_driver(wait=wait, kill=kill)
    sch = scheduler.Scheduler(driver, [realization])
    sch.BATCH_KILLING_INTERVAL = 0.1
    scheduler_task = asyncio.create_task(sch.execute())

    # Wait for the job to start
    await asyncio.wait_for(pre.wait(), timeout=1)

    # Kill all jobs and wait for the scheduler to complete
    await sch.cancel_all_jobs()  # this is equivalent to sch.kill_all_jobs()
    await scheduler_task

    assert pre.is_set()
    assert not post.is_set()
    assert killed


async def test_add_dispatch_information_to_jobs_file(
    storage, tmp_path: Path, mock_driver
):
    test_ee_uri = "tcp://test_ee_uri.com/121/"
    test_ens_id = "test_ens_id121"
    test_ee_token = "test_ee_token_t0k€n121"

    ensemble_size = 10

    ensemble = storage.create_experiment().create_ensemble(
        name="foo", ensemble_size=ensemble_size
    )
    realizations = [
        create_stub_realization(ensemble, tmp_path, iens)
        for iens in range(ensemble_size)
    ]

    sch = scheduler.Scheduler(
        mock_driver(),
        realizations=realizations,
        ens_id=test_ens_id,
    )

    for realization in realizations:
        create_jobs_json(realization)

    sch.add_dispatch_information_to_jobs_file(test_ee_uri, test_ee_token)

    for realization in realizations:
        job_file_path = Path(realization.run_arg.runpath) / "jobs.json"
        content: dict = json.loads(job_file_path.read_text(encoding="utf-8"))
        assert content["ens_id"] == test_ens_id
        assert content["real_id"] == realization.iens
        assert content["dispatch_url"] == test_ee_uri
        assert content["ee_token"] == test_ee_token
        assert type(content["jobList"]) is list
        assert len(content["jobList"]) == 0


@pytest.mark.parametrize("max_submit", [1, 2, 3])
async def test_that_max_submit_was_reached(realization, max_submit, mock_driver):
    retries = 0

    async def init(*args, **kwargs):
        nonlocal retries
        retries += 1

    async def wait():
        return 1

    driver = mock_driver(init=init, wait=wait)
    sch = scheduler.Scheduler(driver, [realization])

    sch._max_submit = max_submit

    assert await sch.execute()
    assert retries == max_submit


async def test_that_max_submit_is_not_reached_on_success(realization, mock_driver):
    retries = 0

    async def init(*args, **kwargs):
        nonlocal retries
        retries += 1

    driver = mock_driver(init=init)
    sch = scheduler.Scheduler(driver, [realization], max_submit=5)

    assert await sch.execute()
    assert retries == 1


@pytest.mark.integration_test
@pytest.mark.timeout(10)
@pytest.mark.flaky(rerun=3)
async def test_max_runtime(realization, mock_driver, caplog):
    wait_started = asyncio.Event()

    async def wait():
        wait_started.set()
        await asyncio.sleep(100)

    realization.max_runtime = 1

    sch = scheduler.Scheduler(mock_driver(wait=wait), [realization])
    sch.BATCH_KILLING_INTERVAL = 0.1
    scheduler_finished_successfully = await asyncio.create_task(sch.execute())
    assert wait_started.is_set()
    assert scheduler_finished_successfully

    timeouteventfound = False
    while not timeouteventfound and not sch._events.empty():
        event = await sch._events.get()
        if type(event) is RealizationTimeout:
            timeouteventfound = True
    assert timeouteventfound

    assert "Realization 0 stopped due to MAX_RUNTIME=1 seconds" in caplog.text


@pytest.mark.integration_test
async def test_no_resubmit_on_max_runtime_kill(realization, mock_driver):
    retries = 0

    async def init(*args, **kwargs):
        nonlocal retries
        retries += 1

    async def wait():
        await asyncio.sleep(100)

    realization.max_runtime = 1

    sch = scheduler.Scheduler(
        mock_driver(init=init, wait=wait), [realization], max_submit=2
    )
    sch.BATCH_KILLING_INTERVAL = 0.1
    assert await sch.execute()

    assert retries == 1


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("max_running", [0, 1, 2, 10])
async def test_max_running(max_running, mock_driver, storage, tmp_path):
    runs: list[bool] = []

    async def wait():
        nonlocal runs
        runs.append(True)
        await asyncio.sleep(0.01)
        runs.append(False)

    # Ensemble size must be larger than max_running to be able
    # to expose issues related to max_running
    ensemble_size = max_running * 3 if max_running > 0 else 10

    ensemble = storage.create_experiment().create_ensemble(
        name="foo", ensemble_size=ensemble_size
    )
    realizations = [
        create_stub_realization(ensemble, tmp_path, iens)
        for iens in range(ensemble_size)
    ]

    sch = scheduler.Scheduler(
        mock_driver(wait=wait), realizations, max_running=max_running
    )

    assert await sch.execute()

    currently_running = 0
    max_running_observed = 0
    for run in runs:
        currently_running += 1 if run else -1
        max_running_observed = max(max_running_observed, currently_running)

    if max_running > 0:
        assert max_running_observed == max_running
    else:
        assert max_running_observed == ensemble_size


@pytest.mark.integration_test
@pytest.mark.timeout(6)
async def test_max_runtime_while_killing(realization, mock_driver):
    wait_started = asyncio.Event()
    now_kill_me = asyncio.Event()

    async def wait():
        # A realization function that lives forever if it was not killed
        wait_started.set()
        await asyncio.sleep(0.1)
        now_kill_me.set()
        await asyncio.sleep(1000)

    async def kill():
        # A kill function that is triggered before the timeout, but finishes
        # after MAX_RUNTIME
        await asyncio.sleep(1)

    realization.max_runtime = 1
    sch = scheduler.Scheduler(mock_driver(wait=wait, kill=kill), [realization])
    sch.BATCH_KILLING_INTERVAL = 0.1

    scheduler_task = asyncio.create_task(sch.execute())

    await now_kill_me.wait()
    await sch.cancel_all_jobs()  # this is equivalent to sch.kill_all_jobs()

    # Sleep until max_runtime must have kicked in:
    await asyncio.sleep(1.1)

    timeouteventfound = False
    while not timeouteventfound and not sch._events.empty():
        event = await sch._events.get()
        if type(event) is RealizationTimeout:
            timeouteventfound = True

    # Assert that a timeout_event is actually emitted, because killing took a
    # long time, and that we should exit normally (asserting no bad things
    # happen just because we have two things killing the realization).

    assert timeouteventfound

    # The result from execute is that we were cancelled, not stopped
    # as if the timeout happened before kill_all_jobs()
    assert not await scheduler_task


@pytest.mark.timeout(6)
async def test_that_job_does_not_retry_when_killed_by_scheduler(
    realization, mock_driver
):
    kill_me = asyncio.Event()
    is_killed = asyncio.Event()

    retries = 0

    async def wait():
        nonlocal retries
        retries += 1
        kill_me.set()
        await is_killed.wait()

    async def kill():
        nonlocal is_killed
        is_killed.set()

    sch = scheduler.Scheduler(
        mock_driver(wait=wait, kill=kill), [realization], max_submit=2
    )
    sch.BATCH_KILLING_INTERVAL = 0.1
    scheduler_task = asyncio.create_task(sch.execute())

    await kill_me.wait()
    await sch.cancel_all_jobs()

    assert not await scheduler_task
    assert retries == 1, "Job was resubmitted after killing"
    event = None
    while not sch._events.empty():
        event = await sch._events.get()
    assert event is not None
    assert type(event) is RealizationFailed


async def test_is_active(mock_driver, realization):
    """The is_active() function is only used by batch_simulation_context.py"""
    realization_started = asyncio.Event()

    async def init(iens, *args, **kwargs):
        realization_started.set()
        await asyncio.sleep(0.001)  # Ensure time to measure activeness

    sch = scheduler.Scheduler(mock_driver(init=init), [realization])

    execute_task = asyncio.create_task(sch.execute())
    assert not sch.is_active()
    await realization_started.wait()
    assert sch.is_active()
    await execute_task
    assert not sch.is_active()


@pytest.mark.timeout(6)
async def test_job_exception_correctly_propagates(mock_driver, realization, caplog):
    pre = asyncio.Event()

    async def wait(iens):
        nonlocal pre
        pre.set()
        raise RuntimeError("Job submission failed!")

    driver = mock_driver(wait=wait)
    sch = scheduler.Scheduler(driver, [realization])

    execute_task = asyncio.create_task(sch.execute())
    await asyncio.wait_for(pre.wait(), timeout=1)
    with pytest.raises(RuntimeError, match="Job submission failed!"):
        await execute_task

    assert sch._jobs[0].state == JobState.FAILED
    assert "Exception in LocalDriver: Job submission failed!" in caplog.text


@pytest.mark.timeout(6)
@pytest.mark.parametrize("should_fail", [True, False])
async def test_that_failed_realization_will_not_be_cancelled(
    should_fail, realization, mock_driver
):
    started = asyncio.Event()
    kill_called = False

    async def wait(iens):
        started.set()
        if should_fail:
            # the job failed with exit code 1
            return 1
        await asyncio.sleep(100)
        return 0

    async def kill(iens):
        nonlocal kill_called
        kill_called = True

    driver = mock_driver(wait=wait, kill=kill)
    sch = scheduler.Scheduler(driver, [realization])
    sch.BATCH_KILLING_INTERVAL = 0.1
    scheduler_task = asyncio.create_task(sch.execute())

    await started.wait()
    await sch.cancel_all_jobs()

    assert not await scheduler_task

    assert kill_called == (not should_fail)


@pytest.mark.timeout(6)
async def test_that_long_running_jobs_were_stopped(
    storage, tmp_path, mock_driver, caplog
):
    killed_iens = []

    async def kill(iens):
        nonlocal killed_iens
        killed_iens.append(iens)

    async def wait(iens):
        # all jobs with iens > 5 will sleep for 10 seconds and should be killed
        if iens < 6:
            await asyncio.sleep(0.1)
        else:
            await asyncio.sleep(10)

    ensemble_size = 10
    ensemble = storage.create_experiment().create_ensemble(
        name="foo", ensemble_size=ensemble_size
    )
    realizations = [
        create_stub_realization(ensemble, tmp_path, iens)
        for iens in range(ensemble_size)
    ]

    sch = scheduler.Scheduler(
        mock_driver(wait=wait, kill=kill),
        realizations,
        max_running=ensemble_size,
    )
    sch.BATCH_KILLING_INTERVAL = 0.1
    assert await sch.execute(min_required_realizations=5)

    stop_long_running_events_found = 0
    while not sch._events.empty():
        event = await sch._events.get()
        if type(event) is RealizationStoppedLongRunning:
            stop_long_running_events_found += 1
    assert stop_long_running_events_found == 4

    assert killed_iens == [6, 7, 8, 9]

    assert "Stopping realization 6 as its running duration" in caplog.text
    assert "Stopping realization 7 as its running duration" in caplog.text
    assert "Stopping realization 8 as its running duration" in caplog.text
    assert "Stopping realization 9 as its running duration" in caplog.text
    assert (
        "is longer than the factor 1.25 multiplied with the average runtime"
        in caplog.text
    )


@pytest.mark.integration_test
@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize(
    "submit_sleep, iens_stride, realization_runtime",
    [(0, 1, 0.1), (0.1, 1, 0.1), (0.1, 1, 0), (0.1, 2, 0)],
)
async def test_submit_sleep(
    submit_sleep,
    iens_stride,  # Gives sparse ensembles when > 1
    realization_runtime,
    storage,
    tmp_path,
    mock_driver,
):
    run_start_times: list[float] = []

    async def wait():
        nonlocal run_start_times
        run_start_times.append(time.time())
        await asyncio.sleep(realization_runtime)

    ensemble_size = 10

    ensemble = storage.create_experiment().create_ensemble(
        name="foo", ensemble_size=ensemble_size * iens_stride
    )
    realizations = [
        create_stub_realization(ensemble, tmp_path, iens * iens_stride)
        for iens in range(ensemble_size)
    ]

    sch = scheduler.Scheduler(
        mock_driver(wait=wait),
        realizations,
        submit_sleep=submit_sleep,
        max_running=0,
    )
    await sch.execute()

    deltas = [
        next_start - start for start, next_start in itertools.pairwise(run_start_times)
    ]
    assert min(deltas) >= submit_sleep * 0.8
    assert max(deltas) <= submit_sleep + 0.1


@pytest.mark.integration_test
@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize(
    "submit_sleep, realization_max_runtime, max_running",
    [
        (0.15, 0.10, 1),
        (0.15, 0.10, 10),
        (0.15, 0.35, 5),
    ],
)
async def test_submit_sleep_with_max_running(
    submit_sleep, realization_max_runtime, max_running, storage, tmp_path, mock_driver
):
    run_start_times: list[float] = []

    async def wait():
        nonlocal run_start_times
        run_start_times.append(time.time())
        # If the realization runtimes are constant, we will never get into
        # the situation where we can start many realizations at the same moment
        runtime = realization_max_runtime * random.random()
        await asyncio.sleep(runtime)

    ensemble_size = 10

    ensemble = storage.create_experiment().create_ensemble(
        name="foo", ensemble_size=ensemble_size
    )
    realizations = [
        create_stub_realization(ensemble, tmp_path, iens)
        for iens in range(ensemble_size)
    ]

    sch = scheduler.Scheduler(
        mock_driver(wait=wait),
        realizations,
        submit_sleep=submit_sleep,
        max_running=max_running,
    )
    await sch.execute()

    deltas = [
        next_start - start for start, next_start in itertools.pairwise(run_start_times)
    ]
    assert min(deltas) >= submit_sleep * 0.8


async def mock_failure(message, *args, **kwargs):
    raise RuntimeError(message)


@pytest.mark.timeout(5)
async def test_that_driver_poll_exceptions_are_propagated(mock_driver, realization):
    driver = mock_driver()
    driver.poll = partial(mock_failure, "Status polling failed")
    sch = scheduler.Scheduler(driver, [realization])
    sch.BATCH_KILLING_INTERVAL = 0.1
    with pytest.raises(RuntimeError, match="Status polling failed"):
        await sch.execute()


@pytest.mark.timeout(5)
async def test_that_publisher_exceptions_are_propagated(
    mock_driver, realization, monkeypatch
):
    driver = mock_driver()
    monkeypatch.setattr(asyncio.Queue, "get", partial(mock_failure, "Publisher failed"))
    sch = scheduler.Scheduler(driver, [realization])
    sch.BATCH_KILLING_INTERVAL = 0.1
    with pytest.raises(RuntimeError, match="Publisher failed"):
        await sch.execute()


@pytest.mark.timeout(5)
async def test_that_process_event_queue_exceptions_are_propagated(
    mock_driver, realization, monkeypatch
):
    monkeypatch.setattr(
        asyncio.Queue, "get", partial(mock_failure, "Processing event queue failed")
    )
    driver = mock_driver()

    sch = scheduler.Scheduler(driver, [realization])
    sch.BATCH_KILLING_INTERVAL = 0.1
    with pytest.raises(RuntimeError, match="Processing event queue failed"):
        await sch.execute()


def test_scheduler_create_lsf_driver():
    queue_name = "foo_queue"
    bsub_cmd = "bar_bsub_cmd"
    bkill_cmd = "foo_bkill_cmd"
    bjobs_cmd = "bar_bjobs_cmd"
    bhist_cmd = "com_bjobs_cmd"
    lsf_resource = "select[cs && x86_64Linux]"
    exclude_host = "host1,host2"
    queue_config_dict = {
        "QUEUE_SYSTEM": QueueSystem.LSF,
        "FORWARD_MODEL": [("FLOW",), ("RMS",)],
        "QUEUE_OPTION": [
            (QueueSystem.LSF, "BSUB_CMD", bsub_cmd),
            (QueueSystem.LSF, "BKILL_CMD", bkill_cmd),
            (QueueSystem.LSF, "BJOBS_CMD", bjobs_cmd),
            (QueueSystem.LSF, "BHIST_CMD", bhist_cmd),
            (QueueSystem.LSF, "LSF_QUEUE", queue_name),
            (QueueSystem.LSF, "LSF_RESOURCE", lsf_resource),
            (QueueSystem.LSF, "EXCLUDE_HOST", exclude_host),
        ],
    }
    queue_config = QueueConfig.from_dict(queue_config_dict)
    driver = create_driver(queue_config.queue_options)
    assert isinstance(driver, LsfDriver)
    assert str(driver._bsub_cmd) == bsub_cmd
    assert str(driver._bkill_cmd) == bkill_cmd
    assert str(driver._bjobs_cmd) == bjobs_cmd
    assert str(driver._bhist_cmd) == bhist_cmd
    assert driver._queue_name == queue_name
    assert driver._resource_requirement == lsf_resource
    assert driver._exclude_hosts == ["host1", "host2"]
    assert driver._project_code == queue_config.queue_options.project_code


def test_scheduler_create_openpbs_driver():
    queue_name = "foo_queue"
    keep_qsub_output = "True"
    cluster_label = "bar_cluster_label"
    job_prefix = "foo_job_prefix"
    qsub_cmd = "bar_qsub_cmd"
    qdel_cmd = "foo_qdel_cmd"
    qstat_cmd = "bar_qstat_cmd"
    queue_config_dict = {
        "QUEUE_SYSTEM": QueueSystem.TORQUE,
        "FORWARD_MODEL": [("FLOW",), ("RMS",)],
        "QUEUE_OPTION": [
            (QueueSystem.TORQUE, "QUEUE", queue_name),
            (QueueSystem.TORQUE, "KEEP_QSUB_OUTPUT", keep_qsub_output),
            (QueueSystem.TORQUE, "CLUSTER_LABEL", cluster_label),
            (QueueSystem.TORQUE, "JOB_PREFIX", job_prefix),
            (QueueSystem.TORQUE, "QSUB_CMD", qsub_cmd),
            (QueueSystem.TORQUE, "QSTAT_CMD", qstat_cmd),
            (QueueSystem.TORQUE, "QDEL_CMD", qdel_cmd),
        ],
    }
    queue_config = QueueConfig.from_dict(queue_config_dict)
    driver = create_driver(queue_config.queue_options)
    assert isinstance(driver, OpenPBSDriver)
    assert driver._queue_name == queue_name
    assert driver._keep_qsub_output is True if keep_qsub_output == "True" else False
    assert driver._cluster_label == cluster_label
    assert driver._job_prefix == job_prefix
    assert str(driver._qsub_cmd) == qsub_cmd
    assert str(driver._qstat_cmd) == qstat_cmd
    assert str(driver._qdel_cmd) == qdel_cmd
    assert driver._project_code == queue_config.queue_options.project_code


async def test_message_present_in_event_on_load_failure(
    realization, mock_driver, monkeypatch
):
    expected_error = "foo bar error"

    async def mocked_load(*args, **kwargs):
        return LoadResult.failure(expected_error)

    monkeypatch.setattr(job, "load_realization_parameters_and_responses", mocked_load)

    sch = scheduler.Scheduler(mock_driver(), [realization])

    async def mock_publisher(*args, **kwargs):
        return

    sch._publisher = mock_publisher

    await sch.execute()

    event = None
    while not sch._events.empty():
        event = await sch._events.get()

    assert expected_error in event.message


async def test_log_warnings_from_forward_model_is_run_once_per_ensemble(
    tmp_path, mock_driver, monkeypatch, storage
):
    ensemble_size = 10
    ensemble = storage.create_experiment().create_ensemble(
        name="foo", ensemble_size=ensemble_size
    )
    realizations = [
        create_stub_realization(ensemble, tmp_path, iens)
        for iens in range(ensemble_size)
    ]
    mocked_stdouterr_parser = AsyncMock(
        return_value=Job.DEFAULT_FILE_VERIFICATION_TIMEOUT
    )
    monkeypatch.setattr(job, "log_warnings_from_forward_model", mocked_stdouterr_parser)
    sch = scheduler.Scheduler(mock_driver(), realizations)
    await sch.execute()
    mocked_stdouterr_parser.assert_called_once()


async def test_schedule_kills_in_batches(
    tmp_path, mock_driver, monkeypatch, storage, caplog
):
    caplog.set_level(logging.INFO)
    ensemble_size = 5
    ensemble = storage.create_experiment().create_ensemble(
        name="foo", ensemble_size=ensemble_size
    )

    realizations = [
        create_stub_realization(ensemble, tmp_path, iens)
        for iens in range(ensemble_size)
    ]

    next_batch_ready_event = asyncio.Event()
    batch_done = asyncio.Event()

    async def mock_wait_for_next_batch(self):
        nonlocal next_batch_ready_event, batch_done
        batch_done.set()
        await next_batch_ready_event.wait()
        # Clear it, so the next round will await it again
        next_batch_ready_event.clear()
        batch_done.clear()

    monkeypatch.setattr(
        scheduler.Scheduler, "_wait_for_next_batch", mock_wait_for_next_batch
    )
    mock_driver = MagicMock()
    mock_driver.kill = AsyncMock()
    sch = scheduler.Scheduler(mock_driver, realizations=realizations)

    first_batch = True
    batch_of_even_realizations = [i for i in range(ensemble_size) if i % 2 == 0]
    batch_of_odd_realizations = [i for i in range(ensemble_size) if i % 2 == 1]
    expected_batches = [batch_of_even_realizations, batch_of_odd_realizations]
    for expected_batch in expected_batches:
        for iens in expected_batch:
            assert not sch._jobs[iens]._was_killed_by_scheduler.is_set()
            await sch.schedule_kill(iens)
            # Should not be set before the batch is actually killed
            assert not sch._jobs[iens]._was_killed_by_scheduler.is_set()

        if first_batch:
            first_batch = False
            # Should not call kill before batch is ready
            mock_driver.kill.assert_not_called()
        else:
            assert not sch._kill_task.done()
        next_batch_ready_event.set()
        await batch_done.wait()
        await asyncio.sleep(0.05)
        assert (
            f"Scheduler killing a batch of {len(expected_batch)} realizations"
            in caplog.text
        )
        mock_driver.kill.assert_called_with(expected_batch)
        for iens in expected_batch:
            assert sch._jobs[iens]._was_killed_by_scheduler.is_set()

    sch._stop_kill_task.set()
    next_batch_ready_event.set()
    if sch._kill_task is not None:
        await sch._kill_task


@pytest.mark.timeout(10)
async def test_batch_killing_runs_batches_in_parallell(
    tmp_path, mock_driver, monkeypatch, storage, caplog
):
    """This is a test that makes sure the batches run in parallell and not in series.
    Previously, this ran in series, and caused jobs to time out as bkill of hundreds
    of jobs at the same time often takes more than 10 seconds.
    """
    caplog.set_level(logging.INFO)
    ensemble_size = 5
    ensemble = storage.create_experiment().create_ensemble(
        name="foo", ensemble_size=ensemble_size
    )

    realizations = [
        create_stub_realization(ensemble, tmp_path, iens)
        for iens in range(ensemble_size)
    ]

    next_batch_ready_event = asyncio.Event()

    async def mock_wait_for_next_batch(self):
        nonlocal next_batch_ready_event
        await next_batch_ready_event.wait()
        # Clear it, so the next round will await it again
        next_batch_ready_event.clear()

    monkeypatch.setattr(
        scheduler.Scheduler, "_wait_for_next_batch", mock_wait_for_next_batch
    )

    driver_kill_call_count = 0
    driver_kill_event = asyncio.Event()

    async def mock_kill(*args, **kwargs) -> None:
        nonlocal driver_kill_call_count, driver_kill_event
        driver_kill_call_count += 1
        await driver_kill_event.wait()

    mock_driver = MagicMock()
    mock_driver.kill = mock_kill
    sch = scheduler.Scheduler(mock_driver, realizations=realizations)

    for odd_or_even_reminder in [0, 1]:
        expected_batch = [
            i for i in range(ensemble_size) if i % 2 == odd_or_even_reminder
        ]
        for iens in expected_batch:
            await sch.schedule_kill(iens)

        next_batch_ready_event.set()
        await asyncio.sleep(0.05)
        assert (
            f"Scheduler killing a batch of {len(expected_batch)} realizations"
            in caplog.text
        )

    assert driver_kill_call_count == 2, (
        "Batch killing did not run multiple batches in parallell"
    )
    # Multiple batches should now be waiting for this event to be set
    driver_kill_event.set()
    # Let batch killing continue and set was_killed_by_scheduler flag
    await asyncio.sleep(0.05)
    for odd_or_even_reminder in [0, 1]:
        expected_batch = [
            i for i in range(ensemble_size) if i % 2 == odd_or_even_reminder
        ]
        for iens in expected_batch:
            assert sch._jobs[iens]._was_killed_by_scheduler.is_set()
    sch._stop_kill_task.set()
    next_batch_ready_event.set()
    assert sch._kill_task is not None
    await sch._kill_task
