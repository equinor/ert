import asyncio
import logging
import shutil
import time
import warnings
from functools import partial, wraps
from math import inf
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from lxml import etree

import ert
from ert.config import ForwardModelStep
from ert.ensemble_evaluator import Realization
from ert.run_arg import RunArg
from ert.run_models.run_model import captured_logs
from ert.scheduler import Scheduler
from ert.scheduler.job import (
    Job,
    JobState,
    log_info_from_exit_file,
    log_warnings_from_forward_model,
)
from ert.storage.load_status import LoadResult
from ert.warnings import PostSimulationWarning


def create_scheduler():
    sch = AsyncMock()
    sch._ens_id = "0"
    sch._events = asyncio.Queue()
    sch.driver = AsyncMock()
    sch._manifest_queue = None
    sch._cancelled = False
    sch._cancelled_by_evaluator = False
    sch.schedule_kill = lambda real: sch.driver.kill([real])
    return sch


@pytest.fixture
def realization():
    run_arg = RunArg(
        run_id="",
        ensemble_storage=MagicMock(),
        iens=0,
        itr=0,
        runpath="test_runpath",
        job_name="test_job",
    )
    realization = Realization(
        iens=run_arg.iens,
        fm_steps=[],
        active=True,
        max_runtime=None,
        run_arg=run_arg,
        num_cpu=1,
        job_script=str(shutil.which("fm_dispatch.py")),
        realization_memory=0,
    )
    return realization


async def assert_scheduler_events(
    scheduler: Scheduler, expected_job_events: list[JobState]
) -> None:
    for expected_job_event in expected_job_events:
        assert scheduler._events.qsize(), (
            f"Expected to find {expected_job_event=} in the event queue"
        )
        event = scheduler._events.get_nowait()
        assert event.queue_event_type == expected_job_event

    # should be no more events
    assert scheduler._events.empty()


@pytest.mark.timeout(5)
async def test_submitted_job_is_cancelled(realization, mock_event):
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    job.WAIT_PERIOD_FOR_TERM_MESSAGE_TO_CANCEL = 0
    job._requested_max_submit = 1
    job.started = mock_event()
    job.returncode.cancel()
    job._was_killed_by_scheduler.set()
    job_task = asyncio.create_task(job._submit_and_run_once(asyncio.BoundedSemaphore()))

    await asyncio.wait_for(job.started._mock_waited, 5)

    job_task.cancel()
    await job_task
    await assert_scheduler_events(
        scheduler,
        [
            JobState.WAITING,
            JobState.SUBMITTING,
            JobState.PENDING,
            JobState.ABORTING,
            JobState.ABORTED,
        ],
    )
    scheduler.driver.kill.assert_called_with([job.iens])
    scheduler.driver.kill.assert_called_once()


@pytest.mark.parametrize(
    "return_code, max_submit, load_result, expected_final_event",
    [
        [0, 1, LoadResult.success(), JobState.COMPLETED],
        [1, 1, LoadResult.success(), JobState.FAILED],
        [0, 1, LoadResult.failure(""), JobState.FAILED],
        [1, 1, LoadResult.failure(""), JobState.FAILED],
        [0, 2, LoadResult.success(), JobState.COMPLETED],
        [1, 2, LoadResult.success(), JobState.FAILED],
        [0, 2, LoadResult.failure(""), JobState.FAILED],
        [1, 2, LoadResult.failure(""), JobState.FAILED],
    ],
)
@pytest.mark.asyncio
async def test_job_run_sends_expected_events(
    return_code: int,
    max_submit: int,
    load_result,
    expected_final_event: JobState,
    realization: Realization,
    monkeypatch,
):
    async def load(**_):
        return load_result

    monkeypatch.setattr(
        ert.scheduler.job, "load_realization_parameters_and_responses", load
    )
    scheduler = create_scheduler()
    monkeypatch.setattr(
        scheduler.driver,
        "read_stdout_and_stderr_files",
        lambda *args: "",
    )
    scheduler.driver._job_error_message_by_iens = {}

    job = Job(scheduler, realization)
    job._verify_checksum = partial(job._verify_checksum, timeout=0)
    job.started.set()

    job_run_task = asyncio.create_task(
        job.run(
            asyncio.Semaphore(), asyncio.Lock(), asyncio.Lock(), max_submit=max_submit
        )
    )

    for attempt in range(max_submit):
        # The execution flow through Job.run() is manipulated through job.returncode
        if attempt < max_submit - 1:
            job.returncode.set_result(1)
            while job.returncode.done():  # noqa: ASYNC110
                # wait until job.run() resets
                # the future after seeing the failure
                await asyncio.sleep(0)
        else:
            job.started.set()
            job.returncode.set_result(return_code)

    await job_run_task

    await assert_scheduler_events(
        scheduler,
        [
            *(
                [
                    JobState.WAITING,
                    JobState.SUBMITTING,
                    JobState.PENDING,
                    JobState.RUNNING,
                    JobState.RESUBMITTING,
                ]
                * max_submit
            )[:-1],
            expected_final_event,
        ],
    )
    scheduler.driver.submit.assert_called_with(
        realization.iens,
        realization.job_script,
        realization.run_arg.runpath,
        name=realization.run_arg.job_name,
        runpath=Path(realization.run_arg.runpath),
        num_cpu=realization.num_cpu,
        realization_memory=0,
    )
    assert scheduler.driver.submit.call_count == max_submit


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.asyncio
async def test_num_cpu_is_propagated_to_driver(realization: Realization):
    realization.num_cpu = 8
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    job_run_task = asyncio.create_task(
        job.run(asyncio.Semaphore(), asyncio.Lock(), asyncio.Lock(), max_submit=1)
    )
    job.started.set()
    job.returncode.set_result(0)
    await job_run_task
    scheduler.driver.submit.assert_called_with(
        realization.iens,
        realization.job_script,
        realization.run_arg.runpath,
        name=realization.run_arg.job_name,
        runpath=Path(realization.run_arg.runpath),
        num_cpu=8,
        realization_memory=0,
    )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.asyncio
async def test_realization_memory_is_propagated_to_driver(realization: Realization):
    realization.realization_memory = 8 * 1024**2
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    job_run_task = asyncio.create_task(
        job.run(asyncio.Semaphore(), asyncio.Lock(), asyncio.Lock(), max_submit=1)
    )
    job.started.set()
    job.returncode.set_result(0)
    await job_run_task
    scheduler.driver.submit.assert_called_with(
        realization.iens,
        realization.job_script,
        realization.run_arg.runpath,
        num_cpu=1,
        realization_memory=8 * 1024**2,
        name=realization.run_arg.job_name,
        runpath=Path(realization.run_arg.runpath),
    )


@pytest.mark.asyncio
async def test_when_waiting_for_disk_sync_times_out_an_error_is_logged(
    realization: Realization, monkeypatch
):
    scheduler = create_scheduler()
    scheduler._manifest_queue = asyncio.Queue()
    file_path = "does/not/exist"
    scheduler.checksum = {
        "test_runpath": {
            "file": {
                "path": file_path,
                "md5sum": "something",
            }
        }
    }
    log_msgs = []
    job = Job(scheduler, realization)

    original_func = job._verify_checksum

    @wraps(original_func)
    async def wrapped_verify_checksum(*args, **kwargs):
        return await partial(original_func, timeout=0)(*args, **kwargs)

    # This allows job.py to access self._verify_checksum.__name__,
    # which doesn't exist on functools.partial()
    job._verify_checksum = wrapped_verify_checksum

    job.started.set()

    with captured_logs(log_msgs, logging.ERROR):
        job_run_task = asyncio.create_task(
            job.run(asyncio.Semaphore(), asyncio.Lock(), asyncio.Lock(), max_submit=1)
        )
        job.started.set()
        job.returncode.set_result(0)
        await job_run_task

    assert "Disk synchronization failed for does/not/exist" in log_msgs


@pytest.mark.asyncio
async def test_when_files_in_manifest_are_not_created_an_error_is_logged(
    realization: Realization, monkeypatch
):
    scheduler = create_scheduler()
    scheduler._manifest_queue = asyncio.Queue()
    file_path = "does/not/exist"
    error = f"Expected file {file_path} not created by forward model!"
    scheduler.checksum = {
        "test_runpath": {
            "file": {
                "path": file_path,
                "error": error,
            }
        }
    }
    log_msgs = []
    job = Job(scheduler, realization)
    job.started.set()

    with captured_logs(log_msgs, logging.ERROR):
        job_run_task = asyncio.create_task(
            job.run(asyncio.Semaphore(), asyncio.Lock(), asyncio.Lock(), max_submit=1)
        )
        job.started.set()
        job.returncode.set_result(0)
        await job_run_task

    assert error in log_msgs


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.asyncio
async def test_when_checksums_do_not_match_a_warning_is_logged(
    realization: Realization,
):
    scheduler = create_scheduler()
    scheduler._manifest_queue = asyncio.Queue()
    file_path = "invalid_md5sum"
    scheduler.checksum = {
        "test_runpath": {
            "file": {
                "path": file_path,
                "md5sum": "something_something_checksum",
            }
        }
    }
    # Create the file
    Path(file_path).write_text("test", encoding="utf-8")

    log_msgs = []
    job = Job(scheduler, realization)
    job.started.set()

    with captured_logs(log_msgs, logging.WARNING):
        job_run_task = asyncio.create_task(
            job.run(asyncio.Semaphore(), asyncio.Lock(), asyncio.Lock(), max_submit=1)
        )
        job.started.set()
        job.returncode.set_result(0)
        await job_run_task

    assert f"File {file_path} checksum verification failed." in log_msgs


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.asyncio
async def test_when_no_checksum_info_is_received_a_warning_is_logged(
    realization: Realization, mocker
):
    scheduler = create_scheduler()
    scheduler._manifest_queue = asyncio.Queue()
    scheduler.checksum = {}
    # Create the file

    log_msgs = []
    job = Job(scheduler, realization)
    job.started.set()

    # Mock asyncio.sleep to fast-forward time
    mocker.patch("asyncio.sleep", return_value=None)

    with captured_logs(log_msgs, logging.WARNING):
        job_run_task = asyncio.create_task(
            job.run(asyncio.Semaphore(), asyncio.Lock(), asyncio.Lock(), max_submit=1)
        )
        job.started.set()
        job.returncode.set_result(0)
        await job_run_task

    assert (
        f"Checksum information not received for {realization.run_arg.runpath}"
        in log_msgs
    )


@pytest.mark.usefixtures("use_tmpdir")
async def test_log_info_from_exit_file(caplog):
    exit_contents = {
        "step": "foojob",
        "reason": "Divizion-by-sero",
        "stderr_file": "somefilename",
        "stderr": "some_error",
    }

    root = etree.Element("root")
    for key, value in exit_contents.items():
        node = etree.Element(key)
        node.text = value
        root.append(node)

    Path("ERROR").write_text(
        str(etree.tostring(root), encoding="utf-8", errors="ignore"), encoding="utf-8"
    )

    log_info_from_exit_file(Path("ERROR"))
    logs = caplog.text
    for magic_string in exit_contents.values():
        assert magic_string in logs


@pytest.mark.usefixtures("use_tmpdir")
async def test_log_info_from_garbled_exit_file(caplog):
    Path("ERROR").write_text("this is not XML", encoding="utf-8")
    log_info_from_exit_file(Path("ERROR"))
    logs = caplog.text
    assert "Realization failed with an invalid XML" in logs
    assert "'this is not XML'" in logs


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "emitted_warning_str, should_be_captured",
    [
        ("FutureWarning: Feature XYZ is deprecated", True),
        ("DeprecationWarning: Feature XYZ is deprecated", True),
        ("Warning: Feature XYZ is deprecated", True),
        ("UserWarning: No metadata, skipping file", True),
        ("PerformanceWarning: DataFrame is highly fragmented.", True),  # Pandas
        (
            "2025-02-12 20:20:25,876 - "
            "semeio.forward_models.design2params.design2params"
            " - WARNING - Design matrix contains empty cells",
            True,
        ),
        (
            "2025-02-12 21:38:06,992:WARNING:fmu.sumo.uploader:"
            "Metadata upload status error exception",
            True,
        ),
        (
            # Example from RMS.stdout
            "Warning! Log KLOGHnet was missing in well 26/3-A-2 H",
            False,
        ),
        (
            # Example from Eclipse
            " @--WARNING  AT TIME    11525.3   DAYS    ( 8-OCT-2058):",
            False,
        ),
    ],
)
async def test_log_warnings_from_forward_model(
    realization, caplog, emitted_warning_str, should_be_captured
):
    start_time = time.time()
    Path(realization.run_arg.runpath).mkdir()
    (Path(realization.run_arg.runpath) / "foo.stdout.0").write_text(
        emitted_warning_str, encoding="utf-8"
    )
    (Path(realization.run_arg.runpath) / "foo.stderr.0").write_text(
        emitted_warning_str, encoding="utf-8"
    )
    realization.fm_steps = [
        ForwardModelStep(
            name="foo",
            executable="foo",
            stdout_file="foo.stdout",
            stderr_file="foo.stderr",
        )
    ]
    if should_be_captured:
        with pytest.warns(PostSimulationWarning):
            await log_warnings_from_forward_model(realization, start_time - 1)
        assert (
            "Realization 0 step foo.0 warned "
            f"1 time(s) in stdout: {emitted_warning_str}" in caplog.text
        )
        assert (
            "Realization 0 step foo.0 warned "
            f"1 time(s) in stderr: {emitted_warning_str}" in caplog.text
        )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            await log_warnings_from_forward_model(realization, start_time - 1)
        assert emitted_warning_str not in caplog.text


@pytest.mark.usefixtures("use_tmpdir")
async def test_old_warnings_are_not_logged(realization, caplog, mocker):
    # Mock asyncio.sleep to fast-forward time
    mocker.patch("asyncio.sleep")

    Path(realization.run_arg.runpath).mkdir()
    (Path(realization.run_arg.runpath) / "foo.stdout.0").write_text(
        "FutureWarning: Feature XYZ is deprecated", encoding="utf-8"
    )
    realization.fm_steps = [
        ForwardModelStep(
            name="foo",
            executable="foo",
            stdout_file="foo.stdout",
            stderr_file="foo.stderr",
        )
    ]
    job_start_time = time.time() + 1  # Pretend that the job started in the future
    await log_warnings_from_forward_model(realization, job_start_time)
    assert "FutureWarning: Feature XYZ" not in caplog.text


@pytest.mark.usefixtures("use_tmpdir")
async def test_long_warning_from_forward_model_is_truncated(
    realization, caplog, mocker
):
    # Mock asyncio.sleep to fast-forward time
    mocker.patch("asyncio.sleep")

    start_time = time.time()
    emitted_warning_str = "FutureWarning: Feature XYZ is deprecated " + " ".join(
        ["foo bar"] * 2000
    )
    Path(realization.run_arg.runpath).mkdir()
    (Path(realization.run_arg.runpath) / "foo.stdout.0").write_text(
        f"{emitted_warning_str}\n{emitted_warning_str}\n{emitted_warning_str}",
        encoding="utf-8",
    )
    realization.fm_steps = [
        ForwardModelStep(
            name="foo",
            executable="foo",
            stdout_file="foo.stdout",
        )
    ]
    with pytest.warns(PostSimulationWarning):
        await log_warnings_from_forward_model(realization, start_time - 1)
    for line in caplog.text.splitlines():
        if "Realization 0 step foo.0 warned" in line:
            assert len(line) <= 2048 + 91


@pytest.mark.usefixtures("use_tmpdir")
async def test_deduplication_of_repeated_warnings_from_forward_model(
    realization, caplog, mocker
):
    # Mock asyncio.sleep to fast-forward time
    mocker.patch("asyncio.sleep")
    start_time = time.time()
    emitted_warning_str = "FutureWarning: Feature XYZ is deprecated"
    Path(realization.run_arg.runpath).mkdir()
    (Path(realization.run_arg.runpath) / "foo.stdout.0").write_text(
        f"{emitted_warning_str}\n{emitted_warning_str}\n{emitted_warning_str}",
        encoding="utf-8",
    )
    realization.fm_steps = [
        ForwardModelStep(
            name="foo",
            executable="foo",
            stdout_file="foo.stdout",
        )
    ]
    with pytest.warns(PostSimulationWarning):
        await log_warnings_from_forward_model(realization, start_time - 1)
    assert (
        f"Realization 0 step foo.0 warned 3 time(s) in stdout: {emitted_warning_str}"
        in caplog.text
    )
    assert caplog.text.count(emitted_warning_str) == 1


async def test_log_warnings_from_forward_model_can_detect_files_being_created_after_delay(  # noqa
    realization, mocker, tmpdir
):
    initial_timeout = Job.DEFAULT_FILE_VERIFICATION_TIMEOUT
    delay = 10

    stdout_file = "foo.stdout"
    stderr_file = "foo.stderr"

    realization.fm_steps = [
        ForwardModelStep(
            name="foo",
            executable="foo",
            stdout_file=stdout_file,
            stderr_file=stderr_file,
        )
    ]

    # Mock Path.exists to return False delay times before returning True
    call_count = [0]

    def true_after_delay(*args, **kwargs):
        call_count[0] += 1
        return call_count[0] > delay

    mocker.patch("asyncio.sleep")
    mocker.patch("pathlib.Path.exists", side_effect=true_after_delay)

    # Mock st_mtime to be infinite
    mock_stat_result = SimpleNamespace(st_mtime=inf)
    mocker.patch("pathlib.Path.stat", return_value=mock_stat_result)

    # Skip reading from file as there is no files to read from
    mocker.patch("pathlib.Path.read_text", return_value="")

    remaining_timeout = await log_warnings_from_forward_model(
        realization, time.time(), initial_timeout
    )
    assert remaining_timeout == initial_timeout - delay


@pytest.mark.parametrize(
    "method_to_test", ["_verify_checksum", "log_warnings_from_forward_model"]
)
async def test_job_logs_timeouts_from_individual_methods(
    realization, mocker, tmpdir, monkeypatch, caplog, method_to_test
):
    scheduler = create_scheduler()
    scheduler.warnings_extracted = False
    scheduler._manifest_queue = asyncio.Queue()
    scheduler.checksum = {}

    job_ = Job(scheduler, realization)

    if method_to_test == "_verify_checksum":
        mocker.patch(
            "ert.scheduler.job.log_warnings_from_forward_model",
            new_callable=AsyncMock,
            return_value=0,
        )
    elif method_to_test == "log_warnings_from_forward_model":
        # Need a fm step to check for files
        realization.fm_steps = [
            ForwardModelStep(
                name="foo",
                executable="foo",
            )
        ]
        mocker.patch.object(Job, "_verify_checksum")

    job_run_task = asyncio.create_task(
        job_.run(asyncio.Semaphore(), asyncio.Lock(), asyncio.Lock(), max_submit=1)
    )
    # Mock asyncio.sleep to fast-forward time
    mocker.patch("asyncio.sleep")

    job_.started.set()
    job_.returncode.set_result(0)

    with caplog.at_level(logging.INFO):
        await asyncio.gather(job_run_task)

    assert (
        f"{method_to_test} timed out after waiting "
        f"{Job.DEFAULT_FILE_VERIFICATION_TIMEOUT} seconds for files" in caplog.text
    )


@pytest.mark.timeout(10)
async def test_killing_job_while_submitting_waits_for_submit_to_be_done(realization):
    """This test is to make sure driver.submit calls are shielded from being
    cancelled if we terminate in the middle of submitting jobs. This was a bug
    where jobs would be submitted, and we had no way of cancelling them as we
    had not gotten the job_id yet.
    """
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    job.WAIT_PERIOD_FOR_TERM_MESSAGE_TO_CANCEL = 0
    job._requested_max_submit = 1
    job._was_killed_by_scheduler.set()
    job_started_submitting = asyncio.Event()
    job_finished_submitting = asyncio.Event()
    job_waited_for_submitting_to_finish = asyncio.Event()

    async def mock_slow_driver_submit(*args, **kwargs) -> None:
        nonlocal job_started_submitting, job_finished_submitting
        job_started_submitting.set()
        await job_finished_submitting.wait()
        job_waited_for_submitting_to_finish.set()

    scheduler.driver.submit = mock_slow_driver_submit
    job_task = asyncio.create_task(job._submit_and_run_once(asyncio.BoundedSemaphore()))

    await asyncio.wait_for(job_started_submitting.wait(), timeout=5)
    job_task.cancel()
    await asyncio.sleep(0)  # Manually yield for task.cancel() to propagate
    # driver.kill should not be called before submit is finished
    scheduler.driver.kill.assert_not_called()
    job_finished_submitting.set()
    await asyncio.wait_for(job_waited_for_submitting_to_finish.wait(), timeout=5)
    await asyncio.wait_for(job_task, timeout=5)

    await assert_scheduler_events(
        scheduler,
        [
            JobState.WAITING,
            JobState.SUBMITTING,
            JobState.ABORTING,
            JobState.ABORTED,
        ],
    )
    scheduler.driver.kill.assert_called_with([job.iens])
    scheduler.driver.kill.assert_called_once()


@pytest.mark.timeout(5)
async def test_killing_job_submitting_waits_for_submit_to_be_done_does_not_hang(
    realization, caplog
):
    """This test is to make sure the shielded driver.submit call will time out
    after a while, as not to cause everything to hang.
    """
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    job.WAIT_PERIOD_FOR_SUBMIT_TO_FINISH = 0.1
    job.WAIT_PERIOD_FOR_TERM_MESSAGE_TO_CANCEL = 0
    job._requested_max_submit = 1
    job._was_killed_by_scheduler.set()
    job_started_submitting = asyncio.Event()
    job_finished_submitting = asyncio.Event()
    submit_was_cancelled = asyncio.Event()

    async def mock_slow_driver_submit(*args, **kwargs) -> None:
        try:
            job_started_submitting.set()
            await asyncio.sleep(5)
            job_finished_submitting.set()
        except asyncio.CancelledError:
            submit_was_cancelled.set()
            raise

    scheduler.driver.submit = mock_slow_driver_submit
    job_task = asyncio.create_task(job._submit_and_run_once(asyncio.BoundedSemaphore()))

    await asyncio.wait_for(job_started_submitting.wait(), timeout=5)
    job_task.cancel()
    await asyncio.sleep(0)  # Manually yield for task.cancel() to propagate
    # driver.kill should not be called before submit is timed out
    scheduler.driver.kill.assert_not_called()
    await asyncio.wait_for(job_task, timeout=5)
    assert not job_finished_submitting.is_set()
    assert submit_was_cancelled.is_set()
    await assert_scheduler_events(
        scheduler,
        [
            JobState.WAITING,
            JobState.SUBMITTING,
            JobState.ABORTING,
            JobState.ABORTED,
        ],
    )
    scheduler.driver.kill.assert_called_with([job.iens])
    scheduler.driver.kill.assert_called_once()
    assert (
        f"Realization {job.iens} was partially submitted and "
        "could not be terminated. Please check manually."
    ) in caplog.text
    assert "Killing it with the driver" not in caplog.text


@pytest.mark.timeout(10)
async def test_killing_job_does_not_hang_on_waiting_for_scheduler_to_kill_in_batches(
    realization, caplog, mock_event
):
    """This test is to make sure the job does not hang forever on waiting for
    the scheduler to run batch killing via driver, and set the flag.
    """
    scheduler = create_scheduler()
    job = Job(scheduler, realization)
    job.WAIT_PERIOD_FOR_SCHEDULER_TO_KILL_IN_BATCH = 0
    job.WAIT_PERIOD_FOR_TERM_MESSAGE_TO_CANCEL = 0
    job._requested_max_submit = 1

    job_task = asyncio.create_task(job._submit_and_run_once(asyncio.BoundedSemaphore()))
    job.started = mock_event()
    await asyncio.wait_for(job.started._mock_waited, 5)
    job_task.cancel()
    await asyncio.wait_for(job_task, timeout=5)
    await assert_scheduler_events(
        scheduler,
        [
            JobState.WAITING,
            JobState.SUBMITTING,
            JobState.PENDING,
            JobState.ABORTING,
            JobState.ABORTED,
        ],
    )
    scheduler.driver.kill.assert_called_once_with([job.iens])
    assert (
        f"Realization {job.iens} did not get confirmation from "
        "scheduler that the batch killing with driver ran successfully."
    ) in caplog.text
