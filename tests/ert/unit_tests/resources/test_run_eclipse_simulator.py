import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from ert.plugins import ErtPluginManager
from tests.ert.utils import SOURCE_DIR

from ._import_from_location import import_from_location

# import run_reservoirsimulator from ert/resources/forward_models
# package-data. This is kept out of the ert package to avoid the
# overhead of importing ert. This is necessary as these may be invoked as a
# subprocess on each realization.


run_reservoirsimulator = import_from_location(
    "run_reservoirsimulator",
    SOURCE_DIR / "src/ert/resources/forward_models/run_reservoirsimulator.py",
)


@pytest.fixture(name="e100_env")
def e100_env(monkeypatch):
    for var, value in (
        ErtPluginManager()
        .get_forward_model_configuration()
        .get("ECLIPSE100", {})
        .items()
    ):
        monkeypatch.setenv(var, value)


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir", "e100_env")
@pytest.mark.requires_eclipse
def test_run_eclipseX00_can_run_eclipse_and_verify(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/EIGHTCELLS.DATA",
        "EIGHTCELLS.DATA",
    )

    erun = run_reservoirsimulator.RunReservoirSimulator(
        simulator="eclipse", version="2019.3", ecl_case="EIGHTCELLS.DATA"
    )
    erun.run_eclipseX00()

    prt_path = Path("EIGHTCELLS.PRT")  # Produced by Eclipse100 binary
    assert prt_path.stat().st_size > 0

    ok_path = Path("EIGHTCELLS.OK")  # Produced by run_eclipseX00()
    assert ok_path.read_text(encoding="utf-8") == "ECLIPSE simulation OK"

    assert len(erun.parse_errors()) == 0

    assert not Path("EIGHTCELLS.h5").exists(), (
        "HDF conversion should not be run by default"
    )


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir", "e100_env")
@pytest.mark.requires_eclipse
def test_ecl100_binary_can_handle_extra_dots_in_casename(source_root):
    """There is code dealing with file extensions in the Eclipse runner
    so it better be tested to work as expected."""
    shutil.copy(
        source_root / "test-data/ert/eclipse/EIGHTCELLS.DATA",
        "EIGHTCELLS.DOT.DATA",
    )

    erun = run_reservoirsimulator.RunReservoirSimulator(
        simulator="eclipse", version="2019.3", ecl_case="EIGHTCELLS.DOT.DATA"
    )
    erun.run_eclipseX00()

    ok_path = Path("EIGHTCELLS.DOT.OK")
    prt_path = Path("EIGHTCELLS.DOT.PRT")

    assert prt_path.stat().st_size > 0
    assert ok_path.read_text(encoding="utf-8") == "ECLIPSE simulation OK"
    assert len(erun.parse_errors()) == 0


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "e100_env")
def test_run_reservoirsimulator_argparse_api(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/EIGHTCELLS.DATA",
        "EIGHTCELLS.DATA",
    )
    run_reservoirsimulator.run_reservoirsimulator(
        ["eclipse", "EIGHTCELLS.DATA", "--version", "2019.3"]
    )

    assert Path("EIGHTCELLS.OK").exists()


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "e100_env")
def test_run_reservorsimulator_on_parallel_deck(source_root):
    # (EIGHTCELLS is too simple to be used with MPI)
    deck = (source_root / "test-data/ert/eclipse/SPE1.DATA").read_text(encoding="utf-8")
    deck = deck.replace("TITLE", "PARALLEL\n  2 /\n\nTITLE")
    Path("SPE1.DATA").write_text(deck, encoding="utf-8")
    run_reservoirsimulator.run_reservoirsimulator(
        ["eclipse", "SPE1.DATA", "--version", "2019.3", "--num-cpu=2"]
    )
    assert Path("SPE1.OK").exists()


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "e100_env")
def test_run_reservoirsimulator_ignores_unrelated_summary_files(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/EIGHTCELLS.DATA",
        "EIGHTCELLS.DATA",
    )
    # Mock files from another existing run
    Path("PREVIOUS_EIGHTCELLS.SMSPEC").touch()
    Path("PREVIOUS_EIGHTCELLS.UNSMRY").touch()
    run_reservoirsimulator.run_reservoirsimulator(
        ["eclipse", "EIGHTCELLS.DATA", "--version", "2019.3"]
    )
    assert Path("EIGHTCELLS.OK").exists()


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "e100_env")
def test_run_reservoirsimulator_when_unsmry_is_ambiguous_with_mpi(source_root):
    deck = (source_root / "test-data/ert/eclipse/SPE1.DATA").read_text(encoding="utf-8")
    deck = deck.replace("TITLE", "PARALLEL\n  2 /\n\nTITLE")
    Path("SPE1.DATA").write_text(deck, encoding="utf-8")
    # Mock files from another existing run
    Path("PREVIOUS_SPE1.SMSPEC").touch()
    Path("PREVIOUS_SPE1.UNSMRY").touch()
    run_reservoirsimulator.run_reservoirsimulator(
        ["eclipse", "SPE1.DATA", "--version", "2019.3", "--num-cpu=2"]
    )
    assert Path("SPE1.OK").exists()


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "e100_env")
def test_run_reservoirsimulator_reports_successful_runs_with_nosim(source_root):
    deck = (source_root / "test-data/ert/eclipse/EIGHTCELLS.DATA").read_text(
        encoding="utf-8"
    )
    deck = deck.replace("TITLE", "NOSIM\n\nTITLE")
    Path("EIGHTCELLS.DATA").write_text(deck, encoding="utf-8")
    run_reservoirsimulator.run_reservoirsimulator(
        ["eclipse", "EIGHTCELLS.DATA", "--version", "2019.3"]
    )
    assert Path("EIGHTCELLS.OK").exists()
    assert not Path("EIGHTCELLS.UNSMRY").exists()


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "e100_env")
def test_run_reservoirsimulator_on_nosim_with_existing_unsmry_file(source_root):
    """This emulates users rerunning Eclipse in an existing runpath"""
    deck = (source_root / "test-data/ert/eclipse/EIGHTCELLS.DATA").read_text(
        encoding="utf-8"
    )
    deck = deck.replace("TITLE", "NOSIM\n\nTITLE")
    Path("EIGHTCELLS.UNSMRY").write_text("", encoding="utf-8")
    Path("EIGHTCELLS.DATA").write_text(deck, encoding="utf-8")
    run_reservoirsimulator.run_reservoirsimulator(
        ["eclipse", "EIGHTCELLS.DATA", "--version", "2019.3"]
    )
    assert Path("EIGHTCELLS.OK").exists()


@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "e100_env")
def test_await_completed_summary_file_does_not_time_out_on_nosim_with_mpi(source_root):
    deck = (source_root / "test-data/ert/eclipse/SPE1.DATA").read_text(encoding="utf-8")
    deck = deck.replace("TITLE", "NOSIM\n\nPARALLEL\n 2 /\n\nTITLE")
    Path("SPE1.DATA").write_text(deck, encoding="utf-8")
    run_reservoirsimulator.run_reservoirsimulator(
        ["eclipse", "SPE1.DATA", "--version", "2019.3", "--num-cpu=2"]
    )
    assert Path("SPE1.OK").exists()
    assert not Path("SPE1.UNSMRY").exists(), (
        "A nosim run should not produce an unsmry file"
    )
    # The timeout will not happen since find_unsmry() returns None.

    # There is no assert on runtime because we cannot predict how long the Eclipse
    # license checkout takes.


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "e100_env")
def test_run_reservoirsimulator_on_nosim_with_mpi_and_existing_unsmry_file(source_root):
    """This emulates users rerunning Eclipse in an existing runpath, with MPI.

    The wait for timeout will not happen, since there are no summary files present.

    This test only effectively asserts that no crash occurs"""
    deck = (source_root / "test-data/ert/eclipse/SPE1.DATA").read_text(encoding="utf-8")
    deck = deck.replace("TITLE", "NOSIM\n\nPARALLEL\n 2 /\n\nTITLE")
    Path("SPE1.UNSMRY").write_text("", encoding="utf-8")
    Path("SPE1.DATA").write_text(deck, encoding="utf-8")
    run_reservoirsimulator.run_reservoirsimulator(
        ["eclipse", "SPE1.DATA", "--version", "2019.3", "--num-cpu=2"]
    )
    # There is no assert on runtime because we cannot predict how long the
    # Eclipse license checkout takes, otherwise we should assert that there
    # is no await for unsmry completion.
    assert Path("SPE1.OK").exists()


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "e100_env")
def test_run_eclipseX00_will_raise_on_deck_errors(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_ERROR.DATA",
        "SPE1_ERROR.DATA",
    )
    erun = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", ecl_case="SPE1_ERROR", version="2019.3"
    )
    with pytest.raises(Exception, match="ERROR"):
        erun.run_eclipseX00()


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "e100_env")
def test_failed_run_gives_nonzero_returncode_and_exception(monkeypatch):
    deck = Path("MOCKED_DECK.DATA")
    deck.touch()
    erun = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", ecl_case=deck.name, version="dummy_version"
    )
    return_value_with_code = mock.MagicMock()
    return_value_with_code.returncode = 1
    monkeypatch.setattr(
        "subprocess.run", mock.MagicMock(return_value=return_value_with_code)
    )
    with pytest.raises(
        # The return code 1 is sometimes translated to 255.
        subprocess.CalledProcessError,
        match=r"Command .*eclrun.* non-zero exit status (1|255)\.$",
    ):
        erun.run_eclipseX00()


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "e100_env")
def test_failing_eclipse_gives_system_exit(monkeypatch):
    deck = Path("MOCKED_DECK.DATA")
    deck.touch()
    return_value_with_code = mock.MagicMock()
    return_value_with_code.returncode = 1
    monkeypatch.setattr(
        "subprocess.run", mock.MagicMock(return_value=return_value_with_code)
    )

    with pytest.raises(SystemExit):
        run_reservoirsimulator.run_reservoirsimulator(
            ["eclipse", deck.name, "--version", "foo"]
        )


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "e100_env")
def test_run_reservoirsimulator_ignores_errors_in_deck_when_requested(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_ERROR.DATA",
        "SPE1_ERROR.DATA",
    )
    run_reservoirsimulator.run_reservoirsimulator(
        ["eclipse", "SPE1_ERROR.DATA", "--version", "2019.3", "--ignore-errors"]
    )


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "e100_env")
def test_flag_needed_to_produce_hdf5_output_with_ecl100(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/EIGHTCELLS.DATA",
        "EIGHTCELLS.DATA",
    )
    run_reservoirsimulator.run_reservoirsimulator(
        ["eclipse", "EIGHTCELLS.DATA", "--version", "2019.3", "--summary-conversion"]
    )
    assert Path("EIGHTCELLS.h5").exists()


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "e100_env")
def test_mpi_is_working_without_run_reservoirsimulator_knowing_it(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_PARALLEL.DATA",
        "SPE1_PARALLEL.DATA",
    )
    assert re.findall(
        r"PARALLEL\s+2", Path("SPE1_PARALLEL.DATA").read_text(encoding="utf-8")
    ), "This test requires a deck needing 2 CPUs"
    run_reservoirsimulator.run_reservoirsimulator(
        ["eclipse", "SPE1_PARALLEL.DATA", "--version", "2019.3"]
    )

    assert Path("SPE1_PARALLEL.PRT").stat().st_size > 0, "Eclipse did not run at all"
    assert Path("SPE1_PARALLEL.MSG").exists(), "No output from MPI process 1"
    assert Path("SPE1_PARALLEL.2.MSG").exists(), "No output from MPI process 2"
    assert not Path("SPE1_PARALLEL.3.MSG").exists(), (
        "There should not be 3 MPI processes"
    )


@pytest.mark.parametrize(
    ("paths_to_touch", "basepath", "expectation"),
    [
        ([], "SPE1", None),
        (["SPE1.UNSMRY"], "SPE1", "SPE1.UNSMRY"),
        (["spe1.unsmry"], "spe1", "spe1.unsmry"),
        (["SPE1.FUNSMRY"], "SPE1", "SPE1.FUNSMRY"),
        (["spe1.funsmry"], "spe1", "spe1.funsmry"),
        (["foo/spe1.unsmry"], "foo/spe1", "foo/spe1.unsmry"),
        (["foo/SPE1.UNSMRY", "SPE1.UNSMRY"], "foo/SPE1", "foo/SPE1.UNSMRY"),
        (["foo/SPE1.UNSMRY", "SPE1.UNSMRY"], "SPE1", "SPE1.UNSMRY"),
        (["EXTRA_SPE1.UNSMRY", "SPE1.UNSMRY"], "SPE1", "SPE1.UNSMRY"),
        (["EXTRA_SPE1.UNSMRY", "SPE1.UNSMRY"], "EXTRA_SPE1", "EXTRA_SPE1.UNSMRY"),
        (["SPE1.UNSMRY", "SPE1.FUNSMRY"], "SPE1", "ValueError"),
        (
            ["SPE1.UNSMRY", "SPE1.unsmry"],
            "SPE1",
            "ValueError" if sys.platform != "darwin" else "SPE1.UNSMRY",
        ),
        (["SPE1.UNSMRY"], "spe1", None),
    ],
)
@pytest.mark.usefixtures("use_tmpdir")
def test_find_unsmry(paths_to_touch, basepath, expectation):
    for path in paths_to_touch:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()
    if expectation == "ValueError":
        with pytest.raises(ValueError, match="Ambiguous reference to unsmry"):
            run_reservoirsimulator.find_unsmry(Path(basepath))
    elif expectation is None:
        assert run_reservoirsimulator.find_unsmry(Path(basepath)) is None
    else:
        assert str(run_reservoirsimulator.find_unsmry(Path(basepath))) == expectation


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
def test_await_completed_summary_file_will_timeout_on_missing_smry():
    assert (
        # Expected wait time is 0.3
        run_reservoirsimulator.await_completed_unsmry_file(
            "SPE1.UNSMRY", max_wait=0.3, poll_interval=0.1
        )
        > 0.3
    )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.requires_eclipse
def test_ecl100_license_error_is_caught():
    prt_error = """\
 @--MESSAGE  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           CHECKING FOR LICENSES

 @--  ERROR  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           LICENSE ERROR  -1 FOR MULTI-SEGMENT WELL OPTION
 @           FEATURE IS INVALID. CHECK YOUR LICENSE FILE AND
 @           THE LICENSE LOG FILE"""
    eclend = """\
 Error summary
 Comments               0
 Warnings               0
 Problems               0
 Errors                 1
 Bugs                   0
 Final cpu       0.00 elapsed       0.08"""

    Path("FOO.PRT").write_text(prt_error + "\n" + eclend, encoding="utf-8")
    Path("FOO.ECLEND").write_text(eclend, encoding="utf-8")
    Path("FOO.DATA").write_text("", encoding="utf-8")

    run = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", "dummyversion", "FOO.DATA"
    )
    with pytest.raises(run_reservoirsimulator.EclError) as exception_info:
        run.assert_eclend()
    assert exception_info.value.failed_due_to_license_problems()


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.requires_eclipse
def test_ecl100_crash_is_not_mistaken_as_license_trouble():
    prt_error = """\
 @--MESSAGE  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           CHECKING FOR LICENSES

 @--  ERROR  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           NON-LINEAR CONVERGENCE FAILURE"""
    eclend = """\
 Error summary
 Comments               0
 Warnings               0
 Problems               0
 Errors                 1
 Bugs                   0
 Final cpu       0.00 elapsed       0.08"""

    Path("FOO.PRT").write_text(prt_error + "\n" + eclend, encoding="utf-8")
    Path("FOO.ECLEND").write_text(eclend, encoding="utf-8")
    Path("FOO.DATA").write_text("", encoding="utf-8")

    run = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", "dummyversion", "FOO.DATA"
    )
    with pytest.raises(run_reservoirsimulator.EclError) as exception_info:
        run.assert_eclend()
    assert not exception_info.value.failed_due_to_license_problems()


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.requires_eclipse
def test_ecl300_license_error_is_caught():
    prt_error = """\
 @--Message:The message service has been activated
 @--Message:Checking for licenses
 @--Message:Checking for licenses
 @--Message:Checking for licenses
 @--Error
 @ ECLIPSE option not allowed in license
 @ Please ask for a new license
 @ Run stopping
            0 Mbytes of storage required
  No active cells found
          249 Mbytes (image size)
"""
    eclend = """\
 Error summary
 Comments               1
 Warnings               2
 Problems               0
 Errors                 1
 Bugs                   0
 Final cpu       0.01 elapsed       0.02
 Emergency stop called from routine   ZSTOPE"""

    Path("FOO.PRT").write_text(prt_error + "\n" + eclend, encoding="utf-8")
    Path("FOO.ECLEND").write_text(eclend, encoding="utf-8")
    Path("FOO.DATA").write_text("", encoding="utf-8")

    run = run_reservoirsimulator.RunReservoirSimulator(
        "e300", "dummyversion", "FOO.DATA"
    )
    with pytest.raises(run_reservoirsimulator.EclError) as exception_info:
        run.assert_eclend()
    assert exception_info.value.failed_due_to_license_problems()


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.requires_eclipse
def test_ecl300_crash_is_not_mistaken_as_license_trouble():
    prt_error = """\
 @--Message:The message service has been activated
 @--Message:Checking for licenses
 @--Message:Checking for licenses
 @--Message:Checking for licenses
 @ Run stopping
            0 Mbytes of storage required
  No active cells found
          249 Mbytes (image size)
"""
    eclend = """\
 Error summary
 Comments               1
 Warnings               2
 Problems               0
 Errors                 1
 Bugs                   0
 Final cpu       0.01 elapsed       0.02
 Emergency stop called from routine   ZSTOPE"""

    Path("FOO.PRT").write_text(prt_error + "\n" + eclend, encoding="utf-8")
    Path("FOO.ECLEND").write_text(eclend, encoding="utf-8")
    Path("FOO.DATA").write_text("", encoding="utf-8")

    run = run_reservoirsimulator.RunReservoirSimulator(
        "e300", "dummyversion", "FOO.DATA"
    )
    with pytest.raises(run_reservoirsimulator.EclError) as exception_info:
        run.assert_eclend()
    assert not exception_info.value.failed_due_to_license_problems()


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.requires_eclipse
def test_license_error_in_slave_is_caught():
    """If a coupled Eclipse model fails in one of the slave runs
    due to license issues, there is no trace of licence in the master PRT file.

    The master PRT file must be trace for the paths to the SLAVE runs
    and then those PRT files must be parsed.

    Note that the name of the DATA file is truncated in the MESSAGE listing
    the slaves.
    """
    Path("slave1").mkdir()
    Path("slave2").mkdir()
    master_prt_error = f"""\
 @--MESSAGE  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           THIS IS JUST A MESSAGE, NOTHING ELSE
 @--MESSAGE  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           STARTING SLAVE SLAVE1   RUNNING EIGHTCEL
 @           ON HOST localhost                        IN DIRECTORY
 @           {os.getcwd()}/slave1
 @--MESSAGE  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           STARTING SLAVE SLAVE2   RUNNING EIGHTCEL
 @           ON HOST localhost                        IN DIRECTORY
 @           {os.getcwd()}/slave2

<various_output>

 @--  ERROR  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           SLAVE RUN SLAVE2   HAS STOPPED WITH AN ERROR CONDITION.
 @           MASTER RUN AND REMAINING SLAVES WILL ALSO STOP.
 """
    master_eclend = """
 Error summary
 Comments               1
 Warnings               1
 Problems               0
 Errors                 1
 Bugs                   0"""

    Path("EIGHTCELLS_MASTER.PRT").write_text(
        master_prt_error + "\n" + master_eclend, encoding="utf-8"
    )
    Path("EIGHTCELLS_MASTER.ECLEND").write_text(master_eclend, encoding="utf-8")

    slave_prt_error = """\
 @--  ERROR  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           LICENSE ERROR -15 FOR MULTI-SEGMENT WELL OPTION
 @           FEATURE IS INVALID. CHECK YOUR LICENSE FILE AND
 @           THE LICENSE LOG FILE
    """
    Path("slave1/EIGHTCELLS_SLAVE.PRT").write_text("", encoding="utf-8")
    Path("slave2/EIGHTCELLS_SLAVE.PRT").write_text(slave_prt_error, encoding="utf-8")
    Path("EIGHTCELLS_MASTER.DATA").write_text("", encoding="utf-8")

    run = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", "dummyversion", "EIGHTCELLS_MASTER.DATA"
    )
    with pytest.raises(run_reservoirsimulator.EclError) as exception_info:
        run.assert_eclend()
    assert exception_info.value.failed_due_to_license_problems()


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.requires_eclipse
def test_crash_in_slave_is_not_mistaken_as_license():
    Path("slave1").mkdir()
    Path("slave2").mkdir()
    master_prt_error = f"""\
 @--MESSAGE  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           THIS IS JUST A MESSAGE, NOTHING ELSE
 @--MESSAGE  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           STARTING SLAVE SLAVE1   RUNNING EIGHTCEL
 @           ON HOST localhost                        IN DIRECTORY
 @           {os.getcwd()}/slave1
 @--MESSAGE  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           STARTING SLAVE SLAVE2   RUNNING EIGHTCEL
 @           ON HOST localhost                        IN DIRECTORY
 @           {os.getcwd()}/slave2

<various_output>

 @--  ERROR  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           SLAVE RUN SLAVE2   HAS STOPPED WITH AN ERROR CONDITION.
 @           MASTER RUN AND REMAINING SLAVES WILL ALSO STOP.
 """
    master_eclend = """
 Error summary
 Comments               1
 Warnings               1
 Problems               0
 Errors                 1
 Bugs                   0"""

    Path("EIGHTCELLS_MASTER.PRT").write_text(
        master_prt_error + "\n" + master_eclend, encoding="utf-8"
    )
    Path("EIGHTCELLS_MASTER.ECLEND").write_text(master_eclend, encoding="utf-8")

    slave_prt_error = """\
 @--  ERROR  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           NON-LINEAR CONVERGENCE FAILURE
    """
    Path("slave1/EIGHTCELLS_SLAVE.PRT").write_text("", encoding="utf-8")
    Path("slave2/EIGHTCELLS_SLAVE.PRT").write_text(slave_prt_error, encoding="utf-8")
    Path("EIGHTCELLS_MASTER.DATA").write_text("", encoding="utf-8")

    run = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", "dummyversion", "EIGHTCELLS_MASTER.DATA"
    )
    with pytest.raises(run_reservoirsimulator.EclError) as exception_info:
        run.assert_eclend()
    assert not exception_info.value.failed_due_to_license_problems()


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.requires_eclipse
def test_too_few_parsed_error_messages_gives_warning():
    prt_error = """\
 @--MESSAGE  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           THIS IS JUST A MESSAGE, NOTHING ELSE"""
    eclend = """
 Error summary
 Comments               0
 Warnings               0
 Problems               0
 Errors                 1
 Bugs                   0"""

    Path("ECLCASE.PRT").write_text(prt_error + "\n" + eclend, encoding="utf-8")
    Path("ECLCASE.ECLEND").write_text(eclend, encoding="utf-8")

    Path("ECLCASE.DATA").write_text("", encoding="utf-8")

    run = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", "dummyversion", "ECLCASE.DATA"
    )
    with pytest.raises(run_reservoirsimulator.EclError) as exception_info:
        run.assert_eclend()
    assert "Warning, mismatch between stated Error count" in str(exception_info.value)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.requires_eclipse
def test_tail_of_prt_file_is_included_when_error_count_inconsistency():
    prt_error = (
        "this_should_not_be_included "
        + "\n" * 10000
        + """
  this_should_be_included

 @--MESSAGE  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           THIS IS JUST A MESSAGE, NOTHING ELSE"""
    )
    eclend = """
 Error summary
 Comments               0
 Warnings               0
 Problems               0
 Errors                 1
 Bugs                   0"""

    Path("ECLCASE.PRT").write_text(prt_error + "\n" + eclend, encoding="utf-8")
    Path("ECLCASE.ECLEND").write_text(eclend, encoding="utf-8")

    Path("ECLCASE.DATA").write_text("", encoding="utf-8")

    run = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", "dummyversion", "ECLCASE.DATA"
    )
    with pytest.raises(run_reservoirsimulator.EclError) as exception_info:
        run.assert_eclend()
    assert "this_should_be_included" in str(exception_info.value)
    assert "this_should_not_be_included" not in str(exception_info.value)


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.requires_eclipse
def test_correct_number_of_parsed_error_messages_gives_no_warning():
    prt_error = """\
 @--  ERROR  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           THIS IS A DUMMY ERROR MESSAGE"""
    eclend = """
 Error summary
 Comments               0
 Warnings               0
 Problems               0
 Errors                 1
 Bugs                   0"""

    Path("ECLCASE.PRT").write_text(prt_error + "\n" + eclend, encoding="utf-8")
    Path("ECLCASE.ECLEND").write_text(eclend, encoding="utf-8")

    Path("ECLCASE.DATA").write_text("", encoding="utf-8")

    run = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", "dummyversion", "ECLCASE.DATA"
    )
    with pytest.raises(run_reservoirsimulator.EclError) as exception_info:
        run.assert_eclend()
    assert "Warning, mismatch between stated Error count" not in str(
        exception_info.value
    )


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.requires_eclipse
def test_slave_started_message_are_not_counted_as_errors():
    prt_error = f"""\
 @--  ERROR  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           THIS IS A DUMMY ERROR MESSAGE

 @--MESSAGE  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           STARTING SLAVE SLAVE1   RUNNING EIGHTCEL
 @           ON HOST localhost                        IN DIRECTORY
 @           {os.getcwd()}/slave1"""
    eclend = """
 Error summary
 Comments               0
 Warnings               0
 Problems               0
 Errors                 1
 Bugs                   0"""

    Path("ECLCASE.PRT").write_text(prt_error + "\n" + eclend, encoding="utf-8")
    Path("ECLCASE.ECLEND").write_text(eclend, encoding="utf-8")

    Path("ECLCASE.DATA").write_text("", encoding="utf-8")

    run = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", "dummyversion", "ECLCASE.DATA"
    )
    with pytest.raises(run_reservoirsimulator.EclError) as exception_info:
        run.assert_eclend()
    assert "Warning, mismatch between stated Error count" not in str(
        exception_info.value
    )


_DUMMY_ERROR_MESSAGE_E100 = """\
 @--  ERROR  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           THIS IS A DUMMY ERROR MESSAGE"""

_DUMMY_ERROR_MESSAGE_MULTIPLE_CPUS_E100 = """\
 @--  ERROR FROM PROCESSOR 1 AT TIME        0.0   DAYS    (21-DEC-2002):
 @           LICENSE FAILURE: ERROR NUMBER IS -4"""

_DUMMY_ERROR_MESSAGE_E300 = """\
 @--Error
 @ ECLIPSE option not allowed in license
 @ Please ask for a new license
 @ Run stopping"""

_DUMMY_SLAVE_STARTED_MESSAGE = """\
 @--MESSAGE  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           STARTING SLAVE SLAVE1   RUNNING EIGHTCEL
 @           ON HOST localhost                        IN DIRECTORY
 @           dummypath/slave1"""


@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.requires_eclipse
@pytest.mark.parametrize(
    ("prt_error", "expected_error_list"),
    [
        (
            _DUMMY_ERROR_MESSAGE_E100,
            [_DUMMY_ERROR_MESSAGE_E100],
        ),
        (
            _DUMMY_ERROR_MESSAGE_MULTIPLE_CPUS_E100,
            [_DUMMY_ERROR_MESSAGE_MULTIPLE_CPUS_E100],
        ),
        (
            _DUMMY_ERROR_MESSAGE_E300,
            [_DUMMY_ERROR_MESSAGE_E300],
        ),
        (
            _DUMMY_SLAVE_STARTED_MESSAGE,
            [_DUMMY_SLAVE_STARTED_MESSAGE],
        ),
        (
            f"""\
 @--MESSAGE  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           THIS IS JUST A MESSAGE, NOTHING ELSE
 @--MESSAGE  AT TIME        0.0   DAYS    ( 1-JAN-2000):
 @           THIS IS JUST A MESSAGE, NOTHING ELSE
{_DUMMY_SLAVE_STARTED_MESSAGE}

<various_output>

{_DUMMY_ERROR_MESSAGE_E100}
 """,
            [_DUMMY_ERROR_MESSAGE_E100, _DUMMY_SLAVE_STARTED_MESSAGE],
        ),
    ],
)
def test_can_parse_errors(prt_error, expected_error_list):
    Path("ECLCASE.PRT").write_text(prt_error + "\n", encoding="utf-8")

    Path("ECLCASE.DATA").write_text("", encoding="utf-8")

    run = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", "dummyversion", "ECLCASE.DATA"
    )
    error_list = run.parse_errors()
    assert error_list == expected_error_list
