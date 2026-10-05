import subprocess
import sys

import pytest


@pytest.mark.slow
def test_that_importing_ert_gui_does_not_crash_with_a_stale_display():
    """Regression test for #14584: a DISPLAY that is set but unreachable must
    not crash when importing ert.gui, even if matplotlib.pyplot was already
    imported elsewhere in the process beforehand.
    """
    result = subprocess.run(
        [sys.executable, "-c", "import matplotlib.pyplot; import ert.gui"],
        env={"DISPLAY": "localhost:99.0"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
