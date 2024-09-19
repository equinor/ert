import inspect
import json
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path
from unittest import mock

import pytest
import yaml

from tests.utils import SOURCE_DIR

from ._import_from_location import import_from_location

# import ecl_config.py and ecl_run from ert/resources/forward-models/res/script
# package-data path which. These are kept out of the ert package to avoid the
# overhead of importing ert. This is necessary as these may be invoked as a
# subprocess on each realization.


ecl_config = import_from_location(
    "ecl_config",
    os.path.join(
        SOURCE_DIR, "src/ert/resources/forward-models/res/script/ecl_config.py"
    ),
)

ecl_run = import_from_location(
    "ecl_run",
    os.path.join(SOURCE_DIR, "src/ert/resources/forward-models/res/script/ecl_run.py"),
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
            "LSB_JOB_ID": None,
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
@mock.patch.dict(os.environ, {"LSB_JOBID": "some-id"})
def test_env(eclrun_conf):
    with open("eclrun", "w", encoding="utf-8") as f, open(
        "DUMMY.DATA", "w", encoding="utf-8"
    ):
        f.write(
            """#!/usr/bin/env python
import os
import json
with open("env.json", "w") as f:
    json.dump(dict(os.environ), f)
"""
        )
    os.chmod("eclrun", os.stat("eclrun").st_mode | stat.S_IEXEC)
    econfig = ecl_config.Ecl100Config()
    eclrun_config = ecl_config.EclrunConfig(econfig, "2019.3")
    erun = ecl_run.EclRun("DUMMY", None, check_status=False)
    with mock.patch.object(
        erun, "_get_run_command", mock.MagicMock(return_value="./eclrun")
    ):
        erun.runEclipse(eclrun_config=eclrun_config)
    with open("env.json", encoding="utf-8") as f:
        run_env = json.load(f)

    eclrun_env = eclrun_conf["eclrun_env"]
    for k, v in eclrun_env.items():
        if v is None:
            assert k not in run_env
            continue

        if k == "PATH":
            assert run_env[k].startswith(v)
        else:
            assert v == run_env[k]


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
@pytest.mark.requires_eclipse
def test_run(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1.DATA"),
        "SPE1.DATA",
    )
    econfig = ecl_config.Ecl100Config()

    erun = ecl_run.EclRun("SPE1.DATA", None)
    erun.runEclipse(eclrun_config=ecl_config.EclrunConfig(econfig, "2019.1"))

    ok_path = os.path.join(erun.runPath(), f"{erun.baseName()}.OK")
    log_path = os.path.join(erun.runPath(), f"{erun.baseName()}.LOG")

    assert os.path.isfile(ok_path)
    assert os.path.isfile(log_path)
    assert os.path.getsize(log_path) > 0

    errors = erun.parseErrors()
    assert len(errors) == 0


@pytest.mark.integration_test
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
@pytest.mark.requires_eclipse
def test_run_new_log_file(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1.DATA"),
        "SPE1.DATA",
    )
    econfig = ecl_config.Ecl100Config()

    erun = ecl_run.EclRun("SPE1.DATA", None)
    erun.runEclipse(eclrun_config=ecl_config.EclrunConfig(econfig, "2019.3"))

    ok_path = os.path.join(erun.runPath(), f"{erun.baseName()}.OK")
    log_path = os.path.join(erun.runPath(), f"{erun.baseName()}.OUT")

    assert os.path.isfile(ok_path)
    assert os.path.isfile(log_path)
    assert os.path.getsize(log_path) > 0

    errors = erun.parseErrors()
    assert len(errors) == 0


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_run_api(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1.DATA"),
        "SPE1.DATA",
    )
    econfig = ecl_config.Ecl100Config()
    ecl_run.run(econfig, ["SPE1.DATA", "--version=2019.3"])

    assert os.path.isfile("SPE1.DATA")


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_failed_run(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1_ERROR.DATA"),
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
def test_failed_run_OK(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1_ERROR.DATA"),
        "SPE1_ERROR.DATA",
    )
    econfig = ecl_config.Ecl100Config()
    ecl_run.run(econfig, ["SPE1_ERROR", "--version=2019.3", "--ignore-errors"])


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_no_hdf5_output_by_default_with_ecl100(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1.DATA"),
        "SPE1.DATA",
    )
    econfig = ecl_config.Ecl100Config()
    # check that by default .h5 file IS NOT produced
    ecl_run.run(econfig, ["SPE1.DATA", "--version=2019.3"])
    assert not os.path.exists("SPE1.h5")


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_flag_to_produce_hdf5_output_with_ecl100(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1.DATA"),
        "SPE1.DATA",
    )
    econfig = ecl_config.Ecl100Config()
    # check that with flag .h5 file IS produced
    ecl_run.run(econfig, ["SPE1.DATA", "--version=2019.3", "--summary-conversion"])
    assert os.path.exists("SPE1.h5")


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_mpi_run(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1_PARALLEL.DATA"),
        "SPE1_PARALLEL.DATA",
    )
    econfig = ecl_config.Ecl100Config()
    ecl_run.run(econfig, ["SPE1_PARALLEL.DATA", "--version=2019.3", "--num-cpu=2"])
    assert os.path.isfile("SPE1_PARALLEL.OUT")
    assert os.path.getsize("SPE1_PARALLEL.OUT") > 0


@pytest.mark.integration_test
@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_summary_block(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1.DATA"),
        "SPE1.DATA",
    )
    econfig = ecl_config.Ecl100Config()
    erun = ecl_run.EclRun("SPE1.DATA", None)
    ret_value = erun.summary_block()
    assert ret_value is None

    erun.runEclipse(eclrun_config=ecl_config.EclrunConfig(econfig, "2019.3"))
    assert erun.summary_block() is not None


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
