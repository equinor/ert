import re
import shutil
from pathlib import Path
from textwrap import dedent

import pytest

from ert.mode_definitions import TEST_RUN_MODE

from .run_cli import run_cli


@pytest.fixture
def eightcells(use_tmpdir, source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/EIGHTCELLS.DATA", "EIGHTCELLS.DATA"
    )


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
