import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import resfo

from tests.ert.utils import SOURCE_DIR

from ._import_from_location import import_from_location

# import ecl_config.py and ecl_run from ert/resources/forward_models
# package-data path which. These are kept out of the ert package to avoid the
# overhead of importing ert. This is necessary as these may be invoked as a
# subprocess on each realization.


run_reservoirsimulator = import_from_location(
    "run_reservoirsimulator",
    SOURCE_DIR / "src/ert/resources/forward_models/run_reservoirsimulator.py",
)


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
@pytest.mark.requires_eclipse
def test_ecl100_binary_can_produce_output(source_root):
    assert os.getenv("SLBSLS_LICENSE_FILE") is not None
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1.DATA",
        "SPE1.DATA",
    )

    erun = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", "2019.3", "SPE1.DATA"
    )
    erun.runEclipseX00()

    ok_path = Path("SPE1.OK")
    prt_path = Path("SPE1.PRT")

    assert ok_path.exists()
    assert prt_path.stat().st_size > 0

    assert len(erun.parseErrors()) == 0

    assert not Path("SPE1.h5").exists(), "HDF conversion should not be run by default"


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir")
def test_runeclrun_argparse_api(source_root):
    # Todo: avoid actually running Eclipse here, use a mock
    # Also test the other command line options.
    assert os.getenv("SLBSLS_LICENSE_FILE") is not None
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1.DATA",
        "SPE1.DATA",
    )
    run_reservoirsimulator.run_reservoirsimulator(["eclipse", "2019.3", "SPE1.DATA"])

    assert Path("SPE1.OK").exists()


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir")
def test_eclrun_will_raise_on_deck_errors(source_root):
    assert os.getenv("SLBSLS_LICENSE_FILE") is not None
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_ERROR.DATA",
        "SPE1_ERROR.DATA",
    )
    erun = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", "2019.3", "SPE1_ERROR"
    )
    with pytest.raises(Exception, match="ERROR"):
        erun.runEclipseX00()


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir")
def test_failed_run_gives_nonzero_returncode_and_exception(monkeypatch):
    erun = run_reservoirsimulator.RunReservoirSimulator(
        "eclipse", "dummy_version", "mocked_anyway.DATA"
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
        erun.runEclipseX00()


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir")
def test_deck_errors_can_be_ignored(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_ERROR.DATA",
        "SPE1_ERROR.DATA",
    )
    run_reservoirsimulator.run_reservoirsimulator(
        ["eclipse", "2019.3", "SPE1.DATA", "--ignore-errors"]
    )


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir")
def test_flag_needed_to_produce_hdf5_output_with_ecl100(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1.DATA",
        "SPE1.DATA",
    )
    run_reservoirsimulator.run_reservoirsimulator(
        ["eclipse", "2019.3", "SPE1.DATA", "--summary-conversion"]
    )
    assert Path("SPE1.h5").exists()


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir")
def test_mpi_run_is_managed_by_system_tool(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_PARALLEL.DATA",
        "SPE1_PARALLEL.DATA",
    )
    assert re.findall(
        r"PARALLEL\s+2", Path("SPE1_PARALLEL.DATA").read_text(encoding="utf-8")
    ), "Test requires a deck needing 2 CPUs"
    run_reservoirsimulator.run_reservoirsimulator(
        ["eclipse", "2019.3", "SPE1_PARALLEL.DATA"]
    )

    assert Path("SPE1_PARALLEL.PRT").stat().st_size > 0, "Eclipse did not run at all"
    assert Path("SPE1_PARALLEL.MSG").exists(), "No output from MPI process 1"
    assert Path("SPE1_PARALLEL.2.MSG").exists(), "No output from MPI process 2"
    assert not Path(
        "SPE1_PARALLEL.3.MSG"
    ).exists(), "There should not be 3 MPI processes"


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
def test_await_completed_summary_file_will_return_asap():
    resfo.write("FOO.UNSMRY", [("INTEHEAD", np.array([1], dtype=np.int32))])
    assert (
        0.01
        # Expected wait time is the poll_interval
        < run_reservoirsimulator.await_completed_unsmry_file(
            "FOO.UNSMRY", max_wait=0.5, poll_interval=0.1
        )
        < 0.4
    )


@pytest.mark.flaky(reruns=5)
@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir")
def test_await_completed_summary_file_will_wait_for_slow_smry():
    # This is a timing test, and has inherent flakiness:
    #  * Reading and writing to the same smry file at the same time
    #    can make the reading throw an exception every time, and can
    #    result in max_wait triggering.
    #  * If the writer thread is starved, two consecutive polls may
    #    yield the same summary length, resulting in premature exit.
    #  * Heavily loaded hardware can make everything go too slow.
    def slow_smry_writer():
        for size in range(10):
            resfo.write(
                "FOO.UNSMRY", (size + 1) * [("INTEHEAD", np.array([1], dtype=np.int32))]
            )
            time.sleep(0.05)

    thread = threading.Thread(target=slow_smry_writer)
    thread.start()
    time.sleep(0.1)  # Let the thread start writing
    assert (
        0.5
        # Minimal wait time is around 0.55
        < run_reservoirsimulator.await_completed_unsmry_file(
            "FOO.UNSMRY", max_wait=4, poll_interval=0.21
        )
        < 2
    )
    thread.join()


@pytest.mark.usefixtures("use_tmpdir")
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
        run.assertECLEND()
    assert exception_info.value.failed_due_to_license_problems()


@pytest.mark.usefixtures("use_tmpdir")
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
        run.assertECLEND()
    assert not exception_info.value.failed_due_to_license_problems()


@pytest.mark.usefixtures("use_tmpdir")
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
        run.assertECLEND()
    assert exception_info.value.failed_due_to_license_problems()


@pytest.mark.usefixtures("use_tmpdir")
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
        run.assertECLEND()
    assert not exception_info.value.failed_due_to_license_problems()


@pytest.mark.usefixtures("use_tmpdir")
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
        run.assertECLEND()
    assert exception_info.value.failed_due_to_license_problems()


@pytest.mark.usefixtures("use_tmpdir")
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
        run.assertECLEND()
    assert not exception_info.value.failed_due_to_license_problems()


@pytest.mark.usefixtures("use_tmpdir")
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
        run.assertECLEND()
    assert "Warning, mismatch between stated Error count" in str(exception_info.value)


@pytest.mark.usefixtures("use_tmpdir")
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
        run.assertECLEND()
    assert "this_should_be_included" in str(exception_info.value)
    assert "this_should_not_be_included" not in str(exception_info.value)


@pytest.mark.usefixtures("use_tmpdir")
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
        run.assertECLEND()
    assert "Warning, mismatch between stated Error count" not in str(
        exception_info.value
    )


@pytest.mark.usefixtures("use_tmpdir")
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
        run.assertECLEND()
    assert "Warning, mismatch between stated Error count" not in str(
        exception_info.value
    )
