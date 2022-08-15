import os
import stat
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore
from typing import Callable, TypedDict

import pytest

from ert._clib.model_callbacks import LoadStatus
from ert._c_wrappers.job_queue import Driver, JobQueueNode, QueueDriverEnum


@dataclass
class RunArg:
    iens: int


class JobConfig(TypedDict):
    job_script: str
    num_cpu: int
    job_name: str
    run_path: str
    ok_callback: Callable
    exit_callback: Callable


def dummy_ok_callback(args):
    (Path(args[1]) / "OK").write_text("success", encoding="utf-8")
    return (LoadStatus.LOAD_SUCCESSFUL, "")


def dummy_exit_callback(args):
    Path("ERROR").write_text("failure", encoding="utf-8")


DUMMY_CONFIG: JobConfig = {
    "job_script": "job_script.py",
    "num_cpu": 1,
    "job_name": "dummy_job_{}",
    "run_path": "dummy_path_{}",
    "ok_callback": dummy_ok_callback,
    "exit_callback": dummy_exit_callback,
}

SIMPLE_SCRIPT = """#!/bin/sh
echo "finished successfully" > STATUS
"""

# This script is susceptible to race conditions. Python works
# better than sh.
FAILING_SCRIPT = """#!/usr/bin/env python
import sys
with open("one_byte_pr_invocation", "a") as f:
    f.write(".")
sys.exit(1)
"""

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


def create_qstat_output(
    state: str, job_id: str = "10001.s034-lcam", bash=False, bashindent=""
):
    assert len(state) == 1
    mocked_output = f"""Job id            Name             User              Time Use S Queue
----------------  ---------------- ----------------  -------- - -----
{job_id: <16}  MyMockedJob      rms                      0 {state:<1} normal   100
"""  # noqa
    if bash:
        return "\n".join(
            [f'{bashindent}echo "{line}"' for line in mocked_output.splitlines()]
        )
    return mocked_output


# A qstat script that works as expected:
MOCK_QSTAT = "#!/bin/sh\n" + create_qstat_output(state="E", bash=True)

# A qstat shell script that will fail on the first invocation, but succeed on
# the second (by persisting its state in the current working directory)
FLAKY_QSTAT = (
    """#!/bin/sh
sleep 1
if [ -s firstwasflaky ]; then
"""
    + create_qstat_output(state="E", bash=True, bashindent="    ")
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


@pytest.mark.parametrize(
    "qsub_script, qstat_script",
    [
        pytest.param(MOCK_QSUB, MOCK_QSTAT, id="none_flaky"),
        pytest.param(FLAKY_QSUB, MOCK_QSTAT, id="flaky_qsub"),
        pytest.param(MOCK_QSUB, FLAKY_QSTAT, id="flaky_qstat"),
        pytest.param(FLAKY_QSUB, FLAKY_QSTAT, id="all_flaky"),
    ],
)
def test_run_torque_job(tmpdir, qsub_script, qstat_script):
    """Verify that the torque driver will succeed in submitting and
    monitoring torque jobs even when the Torque commands qsub and qstat
    are flaky.

    A flaky torque command is a shell script that sometimes but not
    always returns with a non-zero exit code."""
    os.chdir(tmpdir)
    os.putenv("PATH", os.getcwd() + ":" + os.getenv("PATH"))
    driver = Driver(driver_type=QueueDriverEnum.TORQUE_DRIVER, max_running=1)

    script = Path(DUMMY_CONFIG["job_script"])
    script.write_text(SIMPLE_SCRIPT, encoding="utf-8")
    script.chmod(stat.S_IRWXU)

    qsub = Path("qsub")
    qsub.write_text(qsub_script, encoding="utf-8")
    qsub.chmod(stat.S_IRWXU)

    qstat = Path("qstat")
    qstat.write_text(qstat_script, encoding="utf-8")
    qstat.chmod(stat.S_IRWXU)

    job_id = 0
    runpath = Path(DUMMY_CONFIG["run_path"].format(job_id))
    runpath.mkdir()

    job = JobQueueNode(
        job_script=DUMMY_CONFIG["job_script"],
        job_name=DUMMY_CONFIG["job_name"].format(job_id),
        run_path=os.path.realpath(DUMMY_CONFIG["run_path"].format(job_id)),
        num_cpu=1,
        status_file="STATUS",
        ok_file="OK",
        exit_file="ERROR",
        done_callback_function=DUMMY_CONFIG["ok_callback"],
        exit_callback_function=DUMMY_CONFIG["exit_callback"],
        callback_arguments=[
            RunArg(iens=job_id),
            Path(DUMMY_CONFIG["run_path"].format(job_id)).resolve(),
        ],
    )

    pool_sema = BoundedSemaphore(value=2)
    job.run(driver, pool_sema)
    job.wait_for()

    # This file is supposed created by the job that the qsub script points to,
    # but here it is created by the mocked qsub.
    assert Path("job_output").exists()

    # The "done" callback:
    assert (runpath / "OK").read_text(encoding="utf-8") == "success"
