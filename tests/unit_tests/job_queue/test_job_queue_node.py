import os
import stat
from textwrap import dedent
from threading import Timer
from unittest.mock import MagicMock

import hypothesis.strategies as st
import pytest
from hypothesis import given

from ert.config import QueueSystem
from ert.job_queue.driver import Driver
from ert.job_queue.job_queue_node import JobQueueNode
from ert.job_queue.job_status import JobStatus
from ert.job_queue.submit_status import SubmitStatus
from ert.job_queue.thread_status import ThreadStatus
from ert.run_arg import RunArg
from ert.storage import EnsembleAccessor

queue_systems = st.sampled_from(QueueSystem.enums())
job_status = st.sampled_from(JobStatus.enums())
thread_status = st.sampled_from(ThreadStatus)

drivers = st.builds(Driver, queue_systems)

job_script = "mock_job_script"
mock_ensemble_storage = MagicMock(spec=EnsembleAccessor)

runargs = st.builds(
    RunArg,
    iens=st.just(1),
    itr=st.just(0),
    runpath=st.just("."),
    run_id=st.text(),
    job_name=st.text(),
    ensemble_storage=st.just(mock_ensemble_storage),
)
job_queue_nodes = st.builds(
    JobQueueNode,
    job_script=st.just(job_script),
    num_cpu=st.just(1),
    status_file=st.just("STATUS"),
    exit_file=st.just("EXIT"),
    run_arg=runargs,
)


@pytest.fixture(autouse=True)
def setup_mock_queue(tmp_path):
    for c_type, command in [
        ("submit", "bsub"),
        ("submit", "qsub"),
        ("submit", "sbatch"),
        ("job_script", "mock_job_script"),
    ]:
        path = tmp_path / command
        path.write_text(
            dedent(
                f"""\
                #!/bin/bash
                pipe=./{c_type}fifo
                if read line <$pipe; then
                    echo $line
                fi
                """
            ),
            encoding="utf-8",
        )
        path.chmod(stat.S_IEXEC | stat.S_IWUSR | path.stat().st_mode)
    os.mkfifo(tmp_path / "job_scriptfifo", 0o777)
    os.mkfifo(tmp_path / "submitfifo", 0o777)


def next_command_output(command, msg, delay=0.0):
    assert command in ("job_script", "submit")

    def write_to_queue():
        with open(f"./{command}fifo", "w", encoding="utf-8") as fifo:
            fifo.write(msg + "\n")

    Timer(delay, write_to_queue).start()


@given(job_queue_nodes, job_status)
def test_job_status_get_set(job_queue_node, job_status):
    job_queue_node.job_status = job_status
    assert job_queue_node.job_status == job_status


@given(job_queue_nodes, thread_status)
def test_thread_status_get_set(job_queue_node, submit_status):
    job_queue_node.thread_status = submit_status
    assert job_queue_node.thread_status == submit_status


@given(job_queue_nodes)
def test_submit_attempt_is_initially_zero(job_queue_node):
    assert job_queue_node.submit_attempt == 0


@pytest.mark.usefixtures("use_tmpdir")
@given(job_queue_nodes, drivers.filter(lambda d: d.name != "LOCAL"))
def test_when_submit_command_returns_invalid_output_then_submit_fails(
    job_queue_node, driver
):
    next_command_output("submit", "invalid")
    assert job_queue_node.submit(driver) == SubmitStatus.DRIVER_FAIL


@pytest.mark.usefixtures("use_tmpdir")
@given(job_queue_nodes)
def test_submitting_empty_job_on_local_succeeds(job_queue_node):
    driver = Driver(QueueSystem.LOCAL)
    next_command_output("job_script", "")
    assert job_queue_node.submit(driver) == SubmitStatus.OK
    job_queue_node._poll_until_done(driver)
    assert job_queue_node.submit_attempt == 1
