import os
import shutil
import stat
from pathlib import Path
from subprocess import CalledProcessError

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

FLOW_VERSION = "default"


@pytest.fixture
def eightcells(use_tmpdir, source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/EIGHTCELLS.DATA", "EIGHTCELLS.DATA"
    )


@pytest.mark.slow
@pytest.mark.usefixtures("eightcells")
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_flow_can_produce_output():
    run_reservoirsimulator.RunReservoirSimulator(
        "flow", FLOW_VERSION, "EIGHTCELLS.DATA"
    ).run_flow()
    assert Path("EIGHTCELLS.UNSMRY").exists()


def test_flowrun_can_be_bypassed_when_flow_is_available(tmp_path, monkeypatch):
    # Set FLOWRUN_PATH to a path guaranteed not to contain flowrun
    monkeypatch.setenv("FLOWRUN_PATH", str(tmp_path))
    # Add a mocked flow to PATH
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    mocked_flow = Path(tmp_path / "flow")
    mocked_flow.write_text("", encoding="utf-8")
    mocked_flow.chmod(mocked_flow.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "DUMMY.DATA").write_text("", encoding="utf-8")
    runner = run_reservoirsimulator.RunReservoirSimulator(
        "flow", None, str(tmp_path / "DUMMY.DATA")
    )
    assert runner.bypass_flowrun is True


def test_flowrun_cannot_be_bypassed_for_parallel_runs(tmp_path, monkeypatch):
    # Set FLOWRUN_PATH to a path guaranteed not to contain flowrun
    monkeypatch.setenv("FLOWRUN_PATH", str(tmp_path))
    # Add a mocked flow to PATH
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    mocked_flow = Path(tmp_path / "flow")
    mocked_flow.write_text("", encoding="utf-8")
    mocked_flow.chmod(mocked_flow.stat().st_mode | stat.S_IEXEC)

    with pytest.raises(
        RuntimeError, match="MPI runs not supported without a flowrun wrapper"
    ):
        run_reservoirsimulator.RunReservoirSimulator(
            "flow", None, "DUMMY.DATA", num_cpu=2
        )


@pytest.mark.slow
@pytest.mark.usefixtures("eightcells")
@pytest.mark.skipif(not shutil.which("flow"), reason="flow not available")
def test_run_flow_with_no_flowrun(tmp_path, monkeypatch):
    # Set FLOWRUN_PATH to a path guaranteed not to contain flowrun
    monkeypatch.setenv("FLOWRUN_PATH", str(tmp_path))
    run_reservoirsimulator.RunReservoirSimulator(
        "flow", None, "EIGHTCELLS.DATA"
    ).run_flow()
    assert Path("EIGHTCELLS.UNSMRY").exists()


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_flowrunner_will_raise_when_flow_fails(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_ERROR.DATA", "SPE1_ERROR.DATA"
    )
    with pytest.raises(CalledProcessError, match="returned non-zero exit status 1"):
        run_reservoirsimulator.RunReservoirSimulator(
            "flow", FLOW_VERSION, "SPE1_ERROR.DATA"
        ).run_flow()


@pytest.mark.slow
@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_flowrunner_can_ignore_flow_errors(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_ERROR.DATA", "SPE1_ERROR.DATA"
    )
    run_reservoirsimulator.RunReservoirSimulator(
        "flow", FLOW_VERSION, "SPE1_ERROR.DATA", check_status=False
    ).run_flow()


@pytest.mark.slow
@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_flowrunner_will_raise_on_unknown_version():
    Path("DUMMY.DATA").touch()
    with pytest.raises(CalledProcessError):
        run_reservoirsimulator.RunReservoirSimulator(
            "flow", "garbled_version", "DUMMY.DATA"
        ).run_flow()
