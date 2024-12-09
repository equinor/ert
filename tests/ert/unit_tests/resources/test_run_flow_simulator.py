import os
import shutil
from pathlib import Path
from subprocess import CalledProcessError
from unittest import mock

import pytest

from tests.ert.utils import SOURCE_DIR

from ._import_from_location import import_from_location

# import code from ert/forward_models package-data path.
# These are kept out of the ert package to avoid the overhead of
# importing ert. This is necessary as these may be invoked as a subprocess on
# each realization.

run_reservoirsimulator = import_from_location(
    "run_reservoirsimulator",
    SOURCE_DIR / "src/ert/resources/forward_models/run_reservoirsimulator.py",
)

FLOW_VERSION = "daily"


@pytest.mark.integration_test
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_flow_can_produce_output(source_root):
    shutil.copy(source_root / "test-data/ert/eclipse/SPE1.DATA", "SPE1.DATA")
    run_reservoirsimulator.RunReservoirSimulator(
        "flow", FLOW_VERSION, "SPE1.DATA"
    ).runFlow()
    assert Path("SPE1.UNSMRY").exists()


@mock.patch.dict(os.environ, {"FLOWRUN_PATH": ""}, clear=True)
def test_flowrun_can_be_bypassed(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOWRUN_PATH", str(tmp_path))
    print(shutil.which("flowrun"))
    pass


@pytest.mark.integration_test
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_flowrunner_will_raise_when_flow_fails(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_ERROR.DATA", "SPE1_ERROR.DATA"
    )
    with pytest.raises(CalledProcessError, match="returned non-zero exit status 1"):
        run_reservoirsimulator.RunReservoirSimulator(
            "flow", FLOW_VERSION, "SPE1_ERROR.DATA"
        ).runFlow()


@pytest.mark.integration_test
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_flowrunner_will_can_ignore_flow_errors(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_ERROR.DATA", "SPE1_ERROR.DATA"
    )
    run_reservoirsimulator.RunReservoirSimulator(
        "flow", FLOW_VERSION, "SPE1_ERROR.DATA", check_status=False
    ).runFlow()


@pytest.mark.integration_test
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_flowrunner_will_raise_on_unknown_version():
    with pytest.raises(CalledProcessError):
        run_reservoirsimulator.RunReservoirSimulator(
            "flow", "garbled_version", ""
        ).runFlow()


@pytest.mark.integration_test
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_flow_with_parallel_keyword(source_root):
    """This only tests that ERT will be able to start flow on a data deck with
    the PARALLEL keyword present. It does not assert anything regarding whether
    MPI-parallelization will get into play."""
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_PARALLEL.DATA", "SPE1_PARALLEL.DATA"
    )
    run_reservoirsimulator.RunReservoirSimulator(
        "flow", FLOW_VERSION, "SPE1_PARALLEL.DATA"
    ).runFlow()
