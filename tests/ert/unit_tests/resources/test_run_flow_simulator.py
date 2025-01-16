import os
import re
import shutil
import stat
from pathlib import Path
from subprocess import CalledProcessError
from textwrap import dedent

import pytest

from ert.mode_definitions import TEST_RUN_MODE
from tests.ert.utils import SOURCE_DIR

from ...ui_tests.cli.run_cli import run_cli
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


@pytest.fixture()
def eightcells(use_tmpdir, source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/EIGHTCELLS.DATA", "EIGHTCELLS.DATA"
    )


@pytest.mark.integration_test
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


@pytest.mark.integration_test
@pytest.mark.usefixtures("eightcells")
@pytest.mark.skipif(not shutil.which("flow"), reason="flow not available")
def test_run_flow_with_no_flowrun(tmp_path, monkeypatch):
    # Set FLOWRUN_PATH to a path guaranteed not to contain flowrun
    monkeypatch.setenv("FLOWRUN_PATH", str(tmp_path))
    run_reservoirsimulator.RunReservoirSimulator(
        "flow", None, "EIGHTCELLS.DATA"
    ).run_flow()
    assert Path("EIGHTCELLS.UNSMRY").exists()


@pytest.mark.integration_test
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_flowrunner_will_raise_when_flow_fails(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_ERROR.DATA", "SPE1_ERROR.DATA"
    )
    with pytest.raises(CalledProcessError, match="returned non-zero exit status 1"):
        run_reservoirsimulator.RunReservoirSimulator(
            "flow", FLOW_VERSION, "SPE1_ERROR.DATA"
        ).run_flow()


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_flowrunner_will_can_ignore_flow_errors(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_ERROR.DATA", "SPE1_ERROR.DATA"
    )
    run_reservoirsimulator.RunReservoirSimulator(
        "flow", FLOW_VERSION, "SPE1_ERROR.DATA", check_status=False
    ).run_flow()


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_flowrunner_will_raise_on_unknown_version():
    Path("DUMMY.DATA").touch()
    with pytest.raises(CalledProcessError):
        run_reservoirsimulator.RunReservoirSimulator(
            "flow", "garbled_version", "DUMMY.DATA"
        ).run_flow()


@pytest.mark.integration_test
@pytest.mark.usefixtures("eightcells")
@pytest.mark.parametrize("num_cpu", [1, 2])
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_numcpu_maps_to_mpi_processes_with_flow(num_cpu):
    Path("flow.ert").write_text(
        dedent(f"""
    NUM_REALIZATIONS 1
    ECLBASE EIGHTCELLS
    DATA_FILE EIGHTCELLS.DATA
    RUNPATH realization-<IENS>
    NUM_CPU {num_cpu}
    FORWARD_MODEL FLOW
    """).strip(),
        encoding="utf-8",
    )

    run_cli(TEST_RUN_MODE, "--disable-monitoring", "flow.ert")
    flow_stdout = Path("realization-0/FLOW.stdout.0").read_text(encoding="utf-8")
    assert re.search(rf"Number of MPI processes:\s+{num_cpu}", flow_stdout)
    assert re.search(r"Threads per MPI process:\s+1", flow_stdout)


@pytest.mark.integration_test
@pytest.mark.usefixtures("eightcells")
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_user_can_specify_threads_and_oversubscribe_compute_node():
    Path("flow.ert").write_text(
        dedent("""
    NUM_REALIZATIONS 1
    ECLBASE EIGHTCELLS
    DATA_FILE EIGHTCELLS.DATA
    RUNPATH realization-<IENS>
    NUM_CPU 1
    FORWARD_MODEL FLOW(<OPTS>="--threads 2")  -- OPTS is parsed by flowrun
    """).strip(),
        encoding="utf-8",
    )

    run_cli(TEST_RUN_MODE, "--disable-monitoring", "flow.ert")
    flow_stdout = Path("realization-0/FLOW.stdout.0").read_text(encoding="utf-8")
    assert re.search(r"Number of MPI processes:\s+1", flow_stdout)
    assert re.search(r"Threads per MPI process:\s+2", flow_stdout)


@pytest.mark.integration_test
@pytest.mark.usefixtures("eightcells")
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_user_can_specify_threads_and_mpi_processes_and_oversubscribe_compute_node():
    Path("flow.ert").write_text(
        dedent("""
    NUM_REALIZATIONS 1
    ECLBASE EIGHTCELLS
    DATA_FILE EIGHTCELLS.DATA
    RUNPATH realization-<IENS>
    NUM_CPU 1
    FORWARD_MODEL FLOW(<OPTS>="--np 2 --threads 2")  -- OPTS is parsed by flowrun
    """).strip(),
        encoding="utf-8",
    )

    run_cli(TEST_RUN_MODE, "--disable-monitoring", "flow.ert")
    flow_stdout = Path("realization-0/FLOW.stdout.0").read_text(encoding="utf-8")
    assert re.search(r"Number of MPI processes:\s+2", flow_stdout)
    assert re.search(r"Threads per MPI process:\s+2", flow_stdout)


@pytest.mark.integration_test
@pytest.mark.usefixtures("eightcells")
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_setenv_can_be_used_to_set_threads():
    Path("flow.ert").write_text(
        dedent("""
    NUM_REALIZATIONS 1
    ECLBASE EIGHTCELLS
    DATA_FILE EIGHTCELLS.DATA
    RUNPATH realization-<IENS>
    SETENV OMP_NUM_THREADS 2
    NUM_CPU 1
    FORWARD_MODEL FLOW()
    """).strip(),
        encoding="utf-8",
    )

    run_cli(TEST_RUN_MODE, "--disable-monitoring", "flow.ert")
    flow_stdout = Path("realization-0/FLOW.stdout.0").read_text(encoding="utf-8")
    assert re.search(r"Number of MPI processes:\s+1", flow_stdout)
    assert re.search(r"Threads per MPI process:\s+2", flow_stdout)


@pytest.mark.integration_test
@pytest.mark.usefixtures("eightcells")
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_ert_will_fetch_parallel_keyword_and_set_mpi_processes():
    deck = Path("EIGHTCELLS.DATA").read_text(encoding="utf-8")
    assert "PARALLEL" not in deck
    Path("EIGHTCELLS.DATA").write_text(
        deck.replace("DIMENS", "PARALLEL\n 2 /\n\nDIMENS"), encoding="utf-8"
    )
    Path("flow.ert").write_text(
        dedent("""
    NUM_REALIZATIONS 1
    ECLBASE EIGHTCELLS
    DATA_FILE EIGHTCELLS.DATA
    RUNPATH realization-<IENS>
    FORWARD_MODEL FLOW()
    """).strip(),
        encoding="utf-8",
    )

    run_cli(TEST_RUN_MODE, "--disable-monitoring", "flow.ert")
    flow_stdout = Path("realization-0/FLOW.stdout.0").read_text(encoding="utf-8")
    assert re.search(r"Number of MPI processes:\s+2", flow_stdout)
    assert re.search(r"Threads per MPI process:\s+1", flow_stdout)


@pytest.mark.integration_test
@pytest.mark.usefixtures("eightcells")
@pytest.mark.skipif(not shutil.which("flowrun"), reason="flowrun not available")
def test_num_cpu_wins_over_parallel_in_deck():
    deck = Path("EIGHTCELLS.DATA").read_text(encoding="utf-8")
    assert "PARALLEL" not in deck
    Path("EIGHTCELLS.DATA").write_text(
        deck.replace("DIMENS", "PARALLEL\n 4 /\n\nDIMENS"), encoding="utf-8"
    )
    Path("flow.ert").write_text(
        dedent("""
    NUM_REALIZATIONS 1
    NUM_CPU 2
    ECLBASE EIGHTCELLS
    DATA_FILE EIGHTCELLS.DATA
    RUNPATH realization-<IENS>
    FORWARD_MODEL FLOW()
    """).strip(),
        encoding="utf-8",
    )

    run_cli(TEST_RUN_MODE, "--disable-monitoring", "flow.ert")
    flow_stdout = Path("realization-0/FLOW.stdout.0").read_text(encoding="utf-8")
    assert re.search(r"Number of MPI processes:\s+2", flow_stdout)
    assert re.search(r"Threads per MPI process:\s+1", flow_stdout)
