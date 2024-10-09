import inspect
import json
import os
import re
import shutil
import stat
import subprocess
import textwrap
import threading
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import resfo
import yaml

from tests.ert.utils import SOURCE_DIR

from ._import_from_location import import_from_location

# import ecl_config.py and ecl_run from ert/resources/forward_models/res/script
# package-data path which. These are kept out of the ert package to avoid the
# overhead of importing ert. This is necessary as these may be invoked as a
# subprocess on each realization.


ecl_config = import_from_location(
    "ecl_config",
    os.path.join(
        SOURCE_DIR, "src/ert/resources/forward_models/res/script/ecl_config.py"
    ),
)

ecl_run = import_from_location(
    "ecl_run",
    os.path.join(SOURCE_DIR, "src/ert/resources/forward_models/res/script/ecl_run.py"),
)


def find_version(output):
    return re.search(r"flow\s*([\d.]+)", output).group(1)


@pytest.fixture
def eclrun_conf():
    return {
        "eclrun_env": {
            "SLBSLS_LICENSE_FILE": "7321@eclipse-lic-no.statoil.no",
            "ECLPATH": "/prog/res/ecl/grid",
            "PATH": "/prog/res/ecl/grid/macros",
            "F_UFMTENDIAN": "big",
            "LSB_JOBID": None,
        }
    }


@pytest.fixture
def init_eclrun_config(tmp_path, monkeypatch, eclrun_conf):
    with open(tmp_path / "ecl100_config.yml", "w", encoding="utf-8") as f:
        f.write(yaml.dump(eclrun_conf))
    monkeypatch.setenv("ECL100_SITE_CONFIG", "ecl100_config.yml")


def test_get_version_raise():
    econfig = ecl_config.Ecl100Config()
    class_file = inspect.getfile(ecl_config.Ecl100Config)
    class_dir = os.path.dirname(os.path.abspath(class_file))
    msg = os.path.join(class_dir, "ecl100_config.yml")
    with pytest.raises(ValueError, match=msg):
        econfig._get_version(None)


@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_eclrun_will_prepend_path_and_get_env_vars_from_ecl100config(
    eclrun_conf,
):
    # GIVEN a mocked eclrun that only dumps it env variables
    Path("eclrun").write_text(
        textwrap.dedent(
            """\
                #!/usr/bin/env python
                import os
                import json
                with open("env.json", "w") as f:
                    json.dump(dict(os.environ), f)
                """
        ),
        encoding="utf-8",
    )
    os.chmod("eclrun", os.stat("eclrun").st_mode | stat.S_IEXEC)
    Path("DUMMY.DATA").write_text("", encoding="utf-8")
    econfig = ecl_config.Ecl100Config()
    eclrun_config = ecl_config.EclrunConfig(econfig, "dummyversion")
    erun = ecl_run.EclRun("DUMMY", None, check_status=False)
    with mock.patch.object(
        erun, "_get_run_command", mock.MagicMock(return_value="./eclrun")
    ):
        # WHEN eclrun is run
        erun.runEclipse(eclrun_config=eclrun_config)

    # THEN the env provided to eclrun is the same
    # as the env from ecl_config, but PATH has been
    # prepended with the value from ecl_config
    with open("env.json", encoding="utf-8") as f:
        run_env = json.load(f)

    expected_eclrun_env = eclrun_conf["eclrun_env"]
    for key, value in expected_eclrun_env.items():
        if value is None:
            assert key not in run_env
            continue  # Typically LSB_JOBID

        if key == "PATH":
            assert run_env[key].startswith(value)
        else:
            assert value == run_env[key]


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
@pytest.mark.requires_eclipse
def test_ecl100_binary_can_produce_output(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1.DATA",
        "SPE1.DATA",
    )
    econfig = ecl_config.Ecl100Config()

    erun = ecl_run.EclRun("SPE1.DATA", None)
    erun.runEclipse(eclrun_config=ecl_config.EclrunConfig(econfig, "2019.3"))

    ok_path = Path(erun.runPath()) / f"{erun.baseName()}.OK"
    prt_path = Path(erun.runPath()) / f"{erun.baseName()}.PRT"

    assert ok_path.exists()
    assert prt_path.stat().st_size > 0

    assert len(erun.parseErrors()) == 0


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_forward_model_cmd_line_api_works(source_root):
    # ecl_run.run() is the forward model wrapper around ecl_run.runEclipse()
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1.DATA",
        "SPE1.DATA",
    )
    ecl_run.run(ecl_config.Ecl100Config(), ["SPE1.DATA", "--version=2019.3"])
    assert Path("SPE1.OK").exists()


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_eclrun_will_raise_on_deck_errors(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_ERROR.DATA",
        "SPE1_ERROR.DATA",
    )
    econfig = ecl_config.Ecl100Config()
    eclrun_config = ecl_config.EclrunConfig(econfig, "2019.3")
    erun = ecl_run.EclRun("SPE1_ERROR", None)
    with pytest.raises(Exception, match="ERROR"):
        erun.runEclipse(eclrun_config=eclrun_config)


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_failed_run_nonzero_returncode(monkeypatch):
    Path("FOO.DATA").write_text("", encoding="utf-8")
    econfig = ecl_config.Ecl100Config()
    eclrun_config = ecl_config.EclrunConfig(econfig, "2021.3")
    erun = ecl_run.EclRun("FOO.DATA", None)
    monkeypatch.setattr("ecl_run.EclRun.execEclipse", mock.MagicMock(return_value=1))
    with pytest.raises(
        # The return code 1 is sometimes translated to 255.
        subprocess.CalledProcessError,
        match=r"Command .*eclrun.* non-zero exit status (1|255)\.$",
    ):
        erun.runEclipse(eclrun_config=eclrun_config)


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_deck_errors_can_be_ignored(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_ERROR.DATA",
        "SPE1_ERROR.DATA",
    )
    econfig = ecl_config.Ecl100Config()
    ecl_run.run(econfig, ["SPE1_ERROR", "--version=2019.3", "--ignore-errors"])


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_no_hdf5_output_by_default_with_ecl100(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1.DATA",
        "SPE1.DATA",
    )
    econfig = ecl_config.Ecl100Config()
    ecl_run.run(econfig, ["SPE1.DATA", "--version=2019.3"])
    assert not Path("SPE1.h5").exists()


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_flag_needed_to_produce_hdf5_output_with_ecl100(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1.DATA",
        "SPE1.DATA",
    )
    econfig = ecl_config.Ecl100Config()
    ecl_run.run(econfig, ["SPE1.DATA", "--version=2019.3", "--summary-conversion"])
    assert Path("SPE1.h5").exists()


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_mpi_run_is_managed_by_system_tool(source_root):
    shutil.copy(
        source_root / "test-data/ert/eclipse/SPE1_PARALLEL.DATA",
        "SPE1_PARALLEL.DATA",
    )
    assert re.findall(
        r"PARALLEL\s+2", Path("SPE1_PARALLEL.DATA").read_text(encoding="utf-8")
    ), "Test requires a deck needing 2 CPUs"
    econfig = ecl_config.Ecl100Config()
    ecl_run.run(econfig, ["SPE1_PARALLEL.DATA", "--version=2019.3"])

    assert Path("SPE1_PARALLEL.PRT").stat().st_size > 0, "Eclipse did not run at all"
    assert Path("SPE1_PARALLEL.MSG").exists(), "No output from MPI process 1"
    assert Path("SPE1_PARALLEL.2.MSG").exists(), "No output from MPI process 2"
    assert not Path(
        "SPE1_PARALLEL.3.MSG"
    ).exists(), "There should not be 3 MPI processes"


def test_await_completed_summary_file_will_timeout_on_missing_smry():
    assert (
        # Expected wait time is 0.3
        ecl_run.await_completed_unsmry_file(
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
        < ecl_run.await_completed_unsmry_file(
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
        < ecl_run.await_completed_unsmry_file(
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

    run = ecl_run.EclRun("FOO.DATA", "dummysimulatorobject")
    with pytest.raises(ecl_run.EclError) as exception_info:
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

    run = ecl_run.EclRun("FOO.DATA", "dummysimulatorobject")
    with pytest.raises(ecl_run.EclError) as exception_info:
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

    run = ecl_run.EclRun("FOO.DATA", "dummysimulatorobject")
    with pytest.raises(ecl_run.EclError) as exception_info:
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

    run = ecl_run.EclRun("FOO.DATA", "dummysimulatorobject")
    with pytest.raises(ecl_run.EclError) as exception_info:
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

    run = ecl_run.EclRun("EIGHTCELLS_MASTER.DATA", "dummysimulatorobject")
    with pytest.raises(ecl_run.EclError) as exception_info:
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

    run = ecl_run.EclRun("EIGHTCELLS_MASTER.DATA", "dummysimulatorobject")
    with pytest.raises(ecl_run.EclError) as exception_info:
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

    run = ecl_run.EclRun("ECLCASE.DATA", "dummysimulatorobject")
    with pytest.raises(ecl_run.EclError) as exception_info:
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

    run = ecl_run.EclRun("ECLCASE.DATA", "dummysimulatorobject")
    with pytest.raises(ecl_run.EclError) as exception_info:
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

    run = ecl_run.EclRun("ECLCASE.DATA", "dummysimulatorobject")
    with pytest.raises(ecl_run.EclError) as exception_info:
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

    run = ecl_run.EclRun("ECLCASE.DATA", "dummysimulatorobject")
    with pytest.raises(ecl_run.EclError) as exception_info:
        run.assertECLEND()
    assert "Warning, mismatch between stated Error count" not in str(
        exception_info.value
    )
