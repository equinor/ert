import os
import stat
from pathlib import Path
from threading import BoundedSemaphore
from typing import TypedDict
from unittest.mock import MagicMock

import pytest

from ert.config import QueueSystem
from ert.job_queue import Driver, JobQueueNode, JobStatus
from ert.run_arg import RunArg
from ert.storage import EnsembleAccessor


@pytest.fixture(name="temp_working_directory")
def fixture_temp_working_directory(tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)
    yield tmpdir


@pytest.fixture(name="dummy_config")
def fixture_dummy_config():
    return JobConfig(
        num_cpu=1,
        job_name="dummy_job_{}",
        run_path="dummy_path_{}",
    )


class JobConfig(TypedDict):
    num_cpu: int
    job_name: str
    run_path: str


MOCK_QSUB = """#!/bin/sh
echo "torque job submitted" > job_output
echo "$@" >> job_output
echo "10001.s034-lcam"
exit 0
"""

# A qsub shell script that will fail on the first invocation, but succeed on the
# second (by persisting its state in the current working directory)
FLAKY_QSUB = """#!/bin/sh
if [ -s firstwasflaky ]; then
    echo ok > job_output
    echo "10001.s034-lcam"
    exit 0
fi
echo "it was" > firstwasflaky
exit 1
"""


def create_qstat_f_output(
    job_id: str = "10001.s034-lcam",
    state: str = "R",
    exit_status: int = 0,
    bash=False,
    bashindent="",
):
    assert len(state) == 1
    mocked_output = f"Job Id: {job_id}\n"

    if state is not None:
        mocked_output += f"  job_state = {state}\n"
    if exit_status is not None:
        mocked_output += f"  Exit_status = {exit_status}\n"

    if bash:
        return "\n".join(
            [f'{bashindent}echo "{line}"' for line in mocked_output.splitlines()]
        )
    return mocked_output


# A qstat script that works as expected:
MOCK_QSTAT = "#!/bin/sh\n" + create_qstat_f_output(state="E", bash=True)

# A qstat shell script that will fail on the first invocation, but succeed on
# the second (by persisting its state in the current working directory)
FLAKY_QSTAT = (
    """#!/bin/sh
sleep 1
if [ -s firstwasflaky ]; then
"""
    + create_qstat_f_output(state="E", bash=True, bashindent="    ")
    + """
    exit 0
fi
echo "it was" > firstwasflaky
# These stderr messages should be swallowed and muted by driver:
if [ $RANDOM -le 10000 ]; then
    echo "qstat: Invalid credential 10001.s034-lcam" >&2
else
    echo "qstat: Invalid credential" >&2
fi
exit 1
"""
)

# A qstat command that has no output but return code 0 on first invocation
# and nonzero output and return code 0 on second invocation:
LYING_QSTAT = FLAKY_QSTAT.replace("exit 1", "exit 0")


def _deploy_script(scriptname: Path, scripttext: str):
    script = Path(scriptname)
    script.write_text(scripttext, encoding="utf-8")
    script.chmod(stat.S_IRWXU)


def _build_jobqueuenode(job_script, dummy_config: JobConfig, job_id=0):
    runpath = Path(dummy_config["run_path"].format(job_id))
    runpath.mkdir()

    job = JobQueueNode(
        job_script=job_script,
        num_cpu=1,
        status_file="STATUS",
        exit_file="ERROR",
        run_arg=RunArg(
            str(job_id),
            MagicMock(spec=EnsembleAccessor),
            0,
            0,
            os.path.realpath(dummy_config["run_path"].format(job_id)),
            dummy_config["job_name"].format(job_id),
        ),
    )
    return job, runpath


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "qsub_script, qstat_script",
    [
        pytest.param(MOCK_QSUB, MOCK_QSTAT, id="none_flaky"),
        pytest.param(
            MOCK_QSUB.replace(".s034-lcam", ""),
            MOCK_QSTAT.replace(".s034-lcam", ""),
            id="none_flaky_no_namespace",
        ),
        pytest.param(FLAKY_QSUB, MOCK_QSTAT, id="flaky_qsub"),
        pytest.param(MOCK_QSUB, FLAKY_QSTAT, id="flaky_qstat"),
        pytest.param(FLAKY_QSUB, FLAKY_QSTAT, id="all_flaky"),
        pytest.param(MOCK_QSUB, LYING_QSTAT, id="lying_qstat"),
    ],
)
def test_run_torque_job(
    temp_working_directory,
    dummy_config,
    qsub_script,
    qstat_script,
    mock_fm_ok,
    simple_script,
):
    """Verify that the torque driver will succeed in submitting and
    monitoring torque jobs even when the Torque commands qsub and qstat
    are flaky.

    A flaky torque command is a shell script that sometimes but not
    always returns with a non-zero exit code."""

    _deploy_script("qsub", qsub_script)
    _deploy_script("qstat", qstat_script)

    driver = Driver(
        driver_type=QueueSystem.TORQUE,
        options=[("QSTAT_CMD", temp_working_directory / "qstat")],
    )

    job, runpath = _build_jobqueuenode(simple_script, dummy_config)
    job.run(driver, BoundedSemaphore())
    job.wait_for()

    # This file is supposed created by the job that the qsub script points to,
    # but here it is created by the mocked qsub.
    assert Path("job_output").exists()

    # The "done" callback:
    mock_fm_ok.assert_called()


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.parametrize(
    "user_qstat_option, expected_options",
    [("", "-f 10001"), ("-x", "-f -x 10001"), ("-f", "-f -f 10001")],
)
def test_that_torque_driver_passes_options_to_qstat(
    temp_working_directory,
    dummy_config,
    user_qstat_option,
    expected_options,
    simple_script,
):
    """The driver supports setting options to qstat, but the
    hard-coded -f option is always there."""

    _deploy_script("qsub", MOCK_QSUB)
    _deploy_script(
        "qstat",
        "#!/bin/sh\n"
        + create_qstat_f_output(state="E", bash=True)
        + "\n"
        + "echo $@ > qstat_options",
    )

    driver = Driver(
        driver_type=QueueSystem.TORQUE,
        options=[
            ("QSTAT_CMD", temp_working_directory / "qstat"),
            ("QSTAT_OPTIONS", user_qstat_option),
        ],
    )

    job, _runpath = _build_jobqueuenode(simple_script, dummy_config)
    job.run(driver, BoundedSemaphore())
    job.wait_for()

    assert Path("qstat_options").read_text(encoding="utf-8").strip() == expected_options


@pytest.mark.usefixtures("mock_fm_ok", "use_tmpdir")
@pytest.mark.parametrize(
    "job_state, exit_status, expected_status",
    [
        ("E", 0, JobStatus.SUCCESS),
        ("E", 1, JobStatus.EXIT),
        ("F", 0, JobStatus.SUCCESS),
        ("F", 1, JobStatus.EXIT),
        ("C", 0, JobStatus.SUCCESS),
        ("C", 1, JobStatus.EXIT),
    ],
)
def test_torque_job_status_from_qstat_output(
    temp_working_directory,
    dummy_config,
    job_state,
    exit_status,
    expected_status,
    simple_script,
):
    _deploy_script("qsub", MOCK_QSUB)
    _deploy_script(
        "qstat",
        "#!/bin/sh\n"
        + create_qstat_f_output(state=job_state, exit_status=exit_status, bash=True),
    )

    driver = Driver(
        driver_type=QueueSystem.TORQUE,
        options=[("QSTAT_CMD", temp_working_directory / "qstat")],
    )

    job, _runpath = _build_jobqueuenode(simple_script, dummy_config)

    pool_sema = BoundedSemaphore(value=2)
    job.run(driver, pool_sema)
    job.wait_for()
    assert job.status == expected_status
