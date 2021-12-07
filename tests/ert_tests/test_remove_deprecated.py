from pathlib import Path
import pkg_resources
from ert_shared import __version__
import pytest
from packaging import version


def test_remove_workflows():
    package = pkg_resources.resource_filename(
        "ert_shared", "/share/ert/workflows/jobs/internal-gui"
    )
    assert Path(package).is_dir()
    if version.parse(__version__) > version.parse("2.37"):
        pytest.fail(
            "The workflows in ert_shared/share/ert/workflows/jobs/internal-gui have "
            "passed their deprecation period and should be removed along with this test"
        )
