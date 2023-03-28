import os.path
import subprocess
import sys

import pytest

from tests.utils import SOURCE_DIR

forward_models_path = os.path.join(
    SOURCE_DIR, "src/ert/shared/share/ert/forward-models"
)


@pytest.mark.timeout(60)
@pytest.mark.skipif(
    sys.platform.startswith("darwin"), reason="Performance can be flaky"
)
@pytest.mark.parametrize(
    "script",
    [
        os.path.join(forward_models_path, "templating/script/template_render.py"),
        os.path.join(forward_models_path, "res/script/rms.py"),
        os.path.join(forward_models_path, "res/script/ecl_run.py"),
    ],
)
def test_job_runner_startup_overhead(script):
    """
    This test checks that the overhead of running the forward_model_scripts
    does not take too long. If this is failing, its likely an import in these
    scripts is taking too long, for instance `import ert`.
    """
    for _ in range(10):
        subprocess.check_call(
            (
                sys.executable,
                script,
                "-h",
            )
        )
