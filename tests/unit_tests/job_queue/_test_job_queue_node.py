import datetime
import os
import stat
from contextlib import suppress
from pathlib import Path
from textwrap import dedent
from threading import Timer
from typing import Sequence
from unittest.mock import MagicMock

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings

from ert.config import QueueConfig, QueueSystem
from ert.job_queue.driver import Driver
from ert.job_queue.job_queue_node import JobQueueNode
from ert.job_queue.job_status import JobStatus
from ert.job_queue.submit_status import SubmitStatus
from ert.job_queue.thread_status import ThreadStatus
from ert.run_arg import RunArg
from ert.storage import EnsembleAccessor

queue_systems = st.sampled_from(QueueSystem)
job_status = st.sampled_from(JobStatus.enums())
thread_status = st.sampled_from(ThreadStatus)


def make_driver(queue_system: QueueSystem):
    result = Driver(queue_system)
    if queue_system == QueueSystem.TORQUE:
        result.set_option("QSTAT_CMD", "qstat")
    return result


drivers = st.builds(make_driver, queue_systems)

job_script = "mock_job_script"
mock_ensemble_storage = MagicMock(spec=EnsembleAccessor)

runargs = st.builds(
    RunArg,
    iens=st.just(1),
    itr=st.just(0),
    runpath=st.just("."),
    run_id=st.text(),
    job_name=st.just("name"),
    ensemble_storage=st.just(mock_ensemble_storage),
)
job_queue_nodes = st.builds(
    JobQueueNode,
    job_script=st.just(job_script),
    num_cpu=st.just(1),
    run_arg=runargs,
)


def reset_command_queue(tmp_path):
    for command in ("job_script", "submit", "status", "history"):
        with suppress(OSError):
            os.remove(tmp_path / f"{command}fifo")
        os.mkfifo(tmp_path / f"{command}fifo", 0o777)


@pytest.fixture(autouse=True)
def setup_mock_queue(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path), prepend=os.pathsep)
    for c_type, command in [
        ("submit", "bsub"),
        ("submit", "qsub"),
        ("submit", "sbatch"),
        ("status", "bjobs"),
        ("history", "bhist"),
        ("status", "squeue"),
        ("status", "qstat"),
        ("job_script", "mock_job_script"),
    ]:
        path = tmp_path / command
        path.write_text(
            dedent(
                f"""\
                #!/bin/bash
                echo $@ > {c_type}input.txt
                cat ./{c_type}fifo
                """
            ),
            encoding="utf-8",
        )
        path.chmod(stat.S_IEXEC | stat.S_IWUSR | path.stat().st_mode)
        reset_command_queue(tmp_path)


def next_command_output(command, msg, delay=0.0):
    assert command in ("job_script", "submit", "status", "history")

    def write_to_queue():
        with open(f"./{command}fifo", "w", encoding="utf-8") as fifo:
            fifo.write(msg)

    Timer(delay, write_to_queue).start()


# @given(job_queue_nodes, job_status)
# def test_job_status_get_set(job_queue_node, job_status):
#    job_queue_node.job_status = job_status
#    assert job_queue_node.job_status == job_status


# @given(job_queue_nodes, thread_status)
# def test_thread_status_get_set(job_queue_node, submit_status):
#    job_queue_node.thread_status = submit_status
#    assert job_queue_node.thread_status == submit_status


@given(job_queue_nodes)
def test_submit_attempt_is_initially_zero(job_queue_node):
    assert job_queue_node.submit_attempt == 0


@pytest.mark.usefixtures("use_tmpdir")
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(job_queue_nodes, drivers.filter(lambda d: d.name != "LOCAL"))
def test_when_submit_command_returns_invalid_output_then_submit_fails(
    tmp_path, job_queue_node, driver
):
    reset_command_queue(tmp_path)
    next_command_output("submit", "invalid")
    assert job_queue_node.submit(driver) == SubmitStatus.DRIVER_FAIL


def submit_success_output(driver_name: str, jobid: int) -> str:
    if driver_name == "TORQUE":
        return f"{jobid}.hostname"
    if driver_name == "LSF":
        return f"Job <{jobid}> is submitted to default queue <normal>."
    if driver_name == "SLURM":
        return str(jobid)
    return ""


def job_status_as_slurm(status: JobStatus) -> Sequence[str]:
    if status == JobStatus.PENDING:
        return ("PENDING",)
    if status == JobStatus.DONE:
        return ("COMPLETED",)
    if status == JobStatus.RUNNING:
        return ("COMPLETING", "RUNNING", "CONFIGURING")
    if status == JobStatus.EXIT:
        return ("FAILED",)
    if status == JobStatus.IS_KILLED:
        return ("CANCELED",)
    raise ValueError()


def job_status_as_torque(status: JobStatus) -> Sequence[str]:
    if status == JobStatus.PENDING:
        return ("job_state = H", "job_state = Q")
    if status == JobStatus.DONE:
        return ("job_state = E", "job_state = F", "job_state = C")
    if status == JobStatus.RUNNING:
        return ("job_state = R",)
    if status == JobStatus.EXIT:
        return ("Exit_status = -1",)
    raise ValueError()


def status_output(draw, driver_name: str, jobid: int, status: JobStatus) -> str:
    if driver_name == "TORQUE":
        job_status = draw(st.sampled_from(job_status_as_torque(status)))
        return dedent(
            f"""\
        Job Id: {jobid}.s034-lcam
            Job_Name = jobname
            Job_Owner = owner
            queue = normal
            {job_status}
        """
        )
    if driver_name == "LSF":
        return (
            f"JOBID USER STAT QUEUE FROM_HOST EXEC_HOST JOB_NAME SUBMIT_TIME\n"
            f"{jobid} pytest DONE normal host exec_host name {datetime.datetime.now()}\n"
        )
    if driver_name == "SLURM":
        job_status = draw(st.sampled_from(job_status_as_slurm(status)))
        return f"{jobid} {job_status}"
    if driver_name == "LOCAL":
        return ""
    raise ValueError(f"Unknown driver_name {driver_name}")


@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.usefixtures("use_tmpdir")
@given(job_queue_nodes, drivers, st.integers(min_value=1, max_value=2**30), st.data())
def test_submitting_updates_status(tmp_path, job_queue_node, driver, jobid, data):
    reset_command_queue(tmp_path)
    next_command_output("submit", submit_success_output(driver.name, jobid))
    next_command_output("job_script", "")
    next_command_output(
        "status", status_output(data.draw, driver.name, jobid, JobStatus.DONE)
    )
    assert job_queue_node.submit(driver) == SubmitStatus.OK
    assert job_queue_node.submit_attempt == 1
    job_queue_node._poll_until_done(driver)
    assert job_queue_node.queue_status == JobStatus.DONE


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.usefixtures("use_tmpdir")
@given(job_queue_nodes)
def test_that_exclude_host_is_in_the_bsub_resource_request(tmp_path, job_queue_node):
    reset_command_queue(tmp_path)
    next_command_output("submit", submit_success_output("LSF", 1))
    queue_config = QueueConfig(
        job_script=os.path.abspath("script.sh"),
        queue_system=QueueSystem.LSF,
        max_submit=2,
        queue_options={
            QueueSystem.LSF: [
                ("EXCLUDE_HOST", "hostname1, hostname2"),
            ]
        },
    )
    driver = Driver.create_driver(queue_config)
    job_queue_node.submit(driver)
    submitinput = Path("submitinput.txt").read_text(encoding="utf-8")
    assert "hname!='hostname1'" in submitinput
    assert "hname!='hostname2'" in submitinput


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.usefixtures("use_tmpdir")
@given(job_queue_nodes)
def test_that_project_code_is_given_to_bsub(tmp_path, job_queue_node):
    reset_command_queue(tmp_path)
    next_command_output("submit", submit_success_output("LSF", 1))
    queue_config = QueueConfig(
        job_script=os.path.abspath("script.sh"),
        queue_system=QueueSystem.LSF,
        max_submit=2,
        queue_options={QueueSystem.LSF: [("PROJECT_CODE", "project")]},
    )
    driver = Driver.create_driver(queue_config)
    job_queue_node.submit(driver)
    submitinput = Path("submitinput.txt").read_text(encoding="utf-8")
    assert "-P project" in submitinput


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.usefixtures("use_tmpdir")
@given(job_queue_nodes)
def test_that_resource_is_passed_to_bsub(tmp_path, job_queue_node):
    reset_command_queue(tmp_path)
    next_command_output("submit", submit_success_output("LSF", 1))
    queue_config = QueueConfig(
        job_script=os.path.abspath("script.sh"),
        queue_system=QueueSystem.LSF,
        max_submit=2,
        queue_options={
            QueueSystem.LSF: [
                (
                    "LSF_RESOURCE",
                    "bs[yes] select[hname!='host1' && hname!='host2'] span[hosts=1]",
                ),
                ("EXCLUDE_HOST", "host3, host4"),
            ]
        },
    )
    driver = Driver.create_driver(queue_config)
    job_queue_node.submit(driver)
    submitinput = Path("submitinput.txt").read_text(encoding="utf-8")
    assert (
        "-R bs[yes] "
        "select[hname!='host1' && hname!='host2' && hname!='host3' && hname!='host4']"
        " span[hosts=1]" in submitinput
    )


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.usefixtures("use_tmpdir")
@given(job_queue_nodes)
def test_that_queue_name_is_passed_to_bsub(tmp_path, job_queue_node):
    reset_command_queue(tmp_path)
    next_command_output("submit", submit_success_output("LSF", 1))
    queue_config = QueueConfig(
        job_script=os.path.abspath("script.sh"),
        queue_system=QueueSystem.LSF,
        max_submit=2,
        queue_options={QueueSystem.LSF: [("LSF_QUEUE", "name")]},
    )
    driver = Driver.create_driver(queue_config)
    job_queue_node.submit(driver)
    submitinput = Path("submitinput.txt").read_text(encoding="utf-8")
    assert "-q name" in submitinput


@settings(max_examples=1, suppress_health_check=[HealthCheck.function_scoped_fixture])
@pytest.mark.usefixtures("use_tmpdir")
@given(job_queue_nodes, st.data())
@pytest.mark.parametrize(
    "hist_before, hist_after, expected_status",
    [
        (
            "info_line\nheader_line\n 1 user name 1 2 2 2\n",
            "info_line\nheader_line\n 1 user name 6 2 2 2\n",
            JobStatus.PENDING,
        ),
        (
            "info_line\nheader_line\n 1 user name 2 2 2 2\n",
            "info_line\nheader_line\n 1 user name 2 2 6 2\n",
            JobStatus.RUNNING,
        ),
        (
            "info_line\nheader_line\n 1 user name 1 2 3 4\n",
            "info_line\nheader_line\n 1 user name 1 2 3 4\n",
            JobStatus.DONE,
        ),
    ],
)
def test_that_bhist_is_called_if_job_not_in_bstat(
    hist_before, hist_after, expected_status, tmp_path, job_queue_node, data
):
    queue_config = QueueConfig(
        job_script=os.path.abspath("script.sh"),
        queue_system=QueueSystem.LSF,
        max_submit=2,
        queue_options={QueueSystem.LSF: [("LSF_QUEUE", "name")]},
    )
    driver = Driver.create_driver(queue_config)
    reset_command_queue(tmp_path)

    # submit gave id 1
    next_command_output("submit", submit_success_output("LSF", 1))
    job_queue_node.submit(driver)

    # status only found for job 2
    next_command_output("status", status_output(data.draw, "LSF", 2, JobStatus.DONE))
    next_command_output("history", hist_before)
    next_command_output("history", hist_after, delay=4.0)
    assert job_queue_node._poll_queue_status(driver) == expected_status
