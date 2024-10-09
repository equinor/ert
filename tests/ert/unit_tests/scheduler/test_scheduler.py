import asyncio
import json
import random
import shutil
import time
from functools import partial
from pathlib import Path
from typing import List

import pytest

from _ert.events import Id, RealizationFailed, RealizationTimeout
from ert.config import QueueConfig
from ert.constant_filenames import CERT_FILE
from ert.ensemble_evaluator import Realization
from ert.load_status import LoadResult, LoadStatus
from ert.run_arg import RunArg
from ert.scheduler import LsfDriver, OpenPBSDriver, create_driver, job, scheduler
from ert.scheduler.job import JobState


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
        job_script=str(shutil.which("job_dispatch.py")),
        realization_memory=0,
    )
    return realization


async def test_empty(mock_driver):
    sch = scheduler.Scheduler(mock_driver())
    assert await sch.execute() == Id.ENSEMBLE_SUCCEEDED


async def test_single_job(realization, mock_driver):
    future = asyncio.Future()

    async def init(iens, *args, **kwargs):
        future.set_result(iens)

    driver = mock_driver(init=init)

    sch = scheduler.Scheduler(driver, [realization])

    assert await sch.execute() == Id.ENSEMBLE_SUCCEEDED
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
    test_ee_uri = "ws://test_ee_uri.com/121/"
    test_ens_id = "test_ens_id121"
    test_ee_token = "test_ee_token_t0k€n121"
    test_ee_cert = "test_ee_cert121.pem"

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
        ee_uri=test_ee_uri,
        ee_cert=test_ee_cert,
        ee_token=test_ee_token,
    )

    for realization in realizations:
        create_jobs_json(realization)

    sch.add_dispatch_information_to_jobs_file()

    for realization in realizations:
        job_file_path = Path(realization.run_arg.runpath) / "jobs.json"
        cert_file_path = Path(realization.run_arg.runpath) / CERT_FILE
        content: dict = json.loads(job_file_path.read_text(encoding="utf-8"))
        assert content["ens_id"] == test_ens_id
        assert content["real_id"] == realization.iens
        assert content["dispatch_url"] == test_ee_uri
        assert content["ee_token"] == test_ee_token
        assert content["ee_cert_path"] == str(cert_file_path)
        assert type(content["jobList"]) == list and len(content["jobList"]) == 0
        assert cert_file_path.read_text(encoding="utf-8") == test_ee_cert


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

    assert await sch.execute() == Id.ENSEMBLE_SUCCEEDED
    assert retries == max_submit


async def test_that_max_submit_is_not_reached_on_success(realization, mock_driver):
    retries = 0

    async def init(*args, **kwargs):
        nonlocal retries
        retries += 1

    driver = mock_driver(init=init)
    sch = scheduler.Scheduler(driver, [realization], max_submit=5)

    assert await sch.execute() == Id.ENSEMBLE_SUCCEEDED
    assert retries == 1


@pytest.mark.timeout(10)
async def test_max_runtime(realization, mock_driver, caplog):
    wait_started = asyncio.Event()

    async def wait():
        wait_started.set()
        await asyncio.sleep(100)

    realization.max_runtime = 1

    sch = scheduler.Scheduler(mock_driver(wait=wait), [realization])

    result = await asyncio.create_task(sch.execute())
    assert wait_started.is_set()
    assert result == Id.ENSEMBLE_SUCCEEDED

    timeouteventfound = False
    while not timeouteventfound and not sch._events.empty():
        event = await sch._events.get()
        if type(event) is RealizationTimeout:
            timeouteventfound = True
    assert timeouteventfound

    assert "Realization 0 stopped due to MAX_RUNTIME=1 seconds" in caplog.text


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
    result = await sch.execute()
    assert result == Id.ENSEMBLE_SUCCEEDED

    assert retries == 1


@pytest.mark.parametrize("max_running", [0, 1, 2, 10])
async def test_max_running(max_running, mock_driver, storage, tmp_path):
    runs: List[bool] = []

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

    assert await sch.execute() == Id.ENSEMBLE_SUCCEEDED

    currently_running = 0
    max_running_observed = 0
    for run in runs:
        currently_running += 1 if run else -1
        max_running_observed = max(max_running_observed, currently_running)

    if max_running > 0:
        assert max_running_observed == max_running
    else:
        assert max_running_observed == ensemble_size


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
    await scheduler_task

    # The result from execute is that we were cancelled, not stopped
    # as if the timeout happened before kill_all_jobs()
    assert scheduler_task.result() == Id.ENSEMBLE_CANCELLED


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

    scheduler_task = asyncio.create_task(sch.execute())

    await kill_me.wait()
    await sch.cancel_all_jobs()

    await scheduler_task
    assert scheduler_task.result() == Id.ENSEMBLE_CANCELLED
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

    scheduler_task = asyncio.create_task(sch.execute())

    await started.wait()
    await sch.cancel_all_jobs()

    await scheduler_task
    assert scheduler_task.result() == Id.ENSEMBLE_CANCELLED

    assert kill_called == (not should_fail)


@pytest.mark.timeout(6)
async def test_that_long_running_jobs_were_stopped(storage, tmp_path, mock_driver):
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

    assert await sch.execute(min_required_realizations=5) == Id.ENSEMBLE_SUCCEEDED
    assert killed_iens == [6, 7, 8, 9]


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
    run_start_times: List[float] = []

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
        next_start - start
        for start, next_start in zip(run_start_times[:-1], run_start_times[1:])
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
    run_start_times: List[float] = []

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
        next_start - start
        for start, next_start in zip(run_start_times[:-1], run_start_times[1:])
    ]
    assert min(deltas) >= submit_sleep * 0.8


async def mock_failure(message, *args, **kwargs):
    raise RuntimeError(message)


@pytest.mark.timeout(5)
async def test_that_driver_poll_exceptions_are_propagated(mock_driver, realization):
    driver = mock_driver()
    driver.poll = partial(mock_failure, "Status polling failed")

    sch = scheduler.Scheduler(driver, [realization])

    with pytest.raises(RuntimeError, match="Status polling failed"):
        await sch.execute()


@pytest.mark.timeout(5)
async def test_that_publisher_exceptions_are_propagated(
    mock_driver, realization, monkeypatch
):
    driver = mock_driver()
    monkeypatch.setattr(asyncio.Queue, "get", partial(mock_failure, "Publisher failed"))

    sch = scheduler.Scheduler(driver, [realization])
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
        "QUEUE_SYSTEM": "LSF",
        "FORWARD_MODEL": [("FLOW",), ("RMS",)],
        "QUEUE_OPTION": [
            ("LSF", "BSUB_CMD", bsub_cmd),
            ("LSF", "BKILL_CMD", bkill_cmd),
            ("LSF", "BJOBS_CMD", bjobs_cmd),
            ("LSF", "BHIST_CMD", bhist_cmd),
            ("LSF", "LSF_QUEUE", queue_name),
            ("LSF", "LSF_RESOURCE", lsf_resource),
            ("LSF", "EXCLUDE_HOST", exclude_host),
        ],
    }
    queue_config = QueueConfig.from_dict(queue_config_dict)
    driver = create_driver(queue_config)
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
    memory_per_job = "13gb"
    num_nodes = 1
    num_cpus_per_node = 1
    cluster_label = "bar_cluster_label"
    job_prefix = "foo_job_prefix"
    qsub_cmd = "bar_qsub_cmd"
    qdel_cmd = "foo_qdel_cmd"
    qstat_cmd = "bar_qstat_cmd"
    queue_config_dict = {
        "QUEUE_SYSTEM": "TORQUE",
        "FORWARD_MODEL": [("FLOW",), ("RMS",)],
        "QUEUE_OPTION": [
            ("TORQUE", "QUEUE", queue_name),
            ("TORQUE", "KEEP_QSUB_OUTPUT", keep_qsub_output),
            ("TORQUE", "MEMORY_PER_JOB", memory_per_job),
            ("TORQUE", "NUM_NODES", str(num_nodes)),
            ("TORQUE", "NUM_CPUS_PER_NODE", str(num_cpus_per_node)),
            ("TORQUE", "CLUSTER_LABEL", cluster_label),
            ("TORQUE", "JOB_PREFIX", job_prefix),
            ("TORQUE", "QSUB_CMD", qsub_cmd),
            ("TORQUE", "QSTAT_CMD", qstat_cmd),
            ("TORQUE", "QDEL_CMD", qdel_cmd),
        ],
    }
    queue_config = QueueConfig.from_dict(queue_config_dict)
    driver = create_driver(queue_config)
    assert isinstance(driver, OpenPBSDriver)
    assert driver._queue_name == queue_name
    assert driver._keep_qsub_output == True if keep_qsub_output == "True" else False
    assert driver._memory_per_job == memory_per_job
    assert driver._num_nodes == num_nodes
    assert driver._num_cpus_per_node == num_cpus_per_node
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

    async def mocked_forward_model_ok(*args, **kwargs):
        return LoadResult(LoadStatus.LOAD_FAILURE, expected_error)

    monkeypatch.setattr(job, "forward_model_ok", mocked_forward_model_ok)

    sch = scheduler.Scheduler(mock_driver(), [realization])

    async def mock_publisher(*args, **kwargs):
        return

    sch._publisher = mock_publisher

    await sch.execute()

    event = None
    while not sch._events.empty():
        event = await sch._events.get()

    assert expected_error in event.message
