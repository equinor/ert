import subprocess
import sys

import pytest


@pytest.mark.timeout(4)
@pytest.mark.skipif(
    sys.platform.startswith("darwin"), reason="Performance can be flaky"
)
def test_job_runner_startup_overhead():
    """
    This test checks that the overhead of running job_dispatch.py does not take
    too long. If this is failing, its likely an import in _ert_job_runner is
    taking too long (such as e.g. importing anything from ert).
    """
    for _ in range(10):
        subprocess.check_call(
            (
                sys.executable,
                "-m",
                "_ert_job_runner.job_dispatch",
                "-h",
            ),
            stdout=subprocess.DEVNULL,
        )
