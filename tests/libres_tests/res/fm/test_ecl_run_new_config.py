import inspect
import json
import os
import re
import shutil
import stat
from unittest import mock

import pytest
import yaml
from ecl.summary import EclSum

from ert._c_wrappers.fm.ecl import Ecl100Config, EclRun, EclrunConfig, run


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
    with open(tmp_path / "ecl100_config.yml", "w") as f:
        f.write(yaml.dump(eclrun_conf))
    monkeypatch.setenv("ECL100_SITE_CONFIG", "ecl100_config.yml")


def test_get_version_raise():
    ecl_config = Ecl100Config()
    class_file = inspect.getfile(Ecl100Config)
    class_dir = os.path.dirname(os.path.abspath(class_file))
    msg = os.path.join(class_dir, "ecl100_config.yml")
    with pytest.raises(ValueError, match=msg):
        ecl_config._get_version(None)


@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
@mock.patch.dict(os.environ, {"LSB_JOBID": "some-id"})
def test_env(eclrun_conf):
    with open("eclrun", "w") as f, open("DUMMY.DATA", "w"):
        f.write(
            """#!/usr/bin/env python
import os
import json
with open("env.json", "w") as f:
    json.dump(dict(os.environ), f)
"""
        )
    os.chmod("eclrun", os.stat("eclrun").st_mode | stat.S_IEXEC)
    ecl_config = Ecl100Config()
    eclrun_config = EclrunConfig(ecl_config, "2019.3")
    ecl_run = EclRun("DUMMY", None, check_status=False)
    with mock.patch.object(
        ecl_run, "_get_run_command", mock.MagicMock(return_value="./eclrun")
    ):
        ecl_run.runEclipse(eclrun_config=eclrun_config)
    with open("env.json") as f:
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


@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
@pytest.mark.requires_eclipse
def test_run(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1.DATA"),
        "SPE1.DATA",
    )
    ecl_config = Ecl100Config()

    ecl_run = EclRun("SPE1.DATA", None)
    ecl_run.runEclipse(eclrun_config=EclrunConfig(ecl_config, "2019.1"))

    ok_path = os.path.join(ecl_run.runPath(), f"{ecl_run.baseName()}.OK")
    log_path = os.path.join(ecl_run.runPath(), f"{ecl_run.baseName()}.LOG")

    assert os.path.isfile(ok_path)
    assert os.path.isfile(log_path)
    assert os.path.getsize(log_path) > 0

    errors = ecl_run.parseErrors()
    assert len(errors) == 0


@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
@pytest.mark.requires_eclipse
def test_run_new_log_file(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1.DATA"),
        "SPE1.DATA",
    )
    ecl_config = Ecl100Config()

    ecl_run = EclRun("SPE1.DATA", None)
    ecl_run.runEclipse(eclrun_config=EclrunConfig(ecl_config, "2019.3"))

    ok_path = os.path.join(ecl_run.runPath(), f"{ecl_run.baseName()}.OK")
    log_path = os.path.join(ecl_run.runPath(), f"{ecl_run.baseName()}.OUT")

    assert os.path.isfile(ok_path)
    assert os.path.isfile(log_path)
    assert os.path.getsize(log_path) > 0

    errors = ecl_run.parseErrors()
    assert len(errors) == 0


@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_run_api(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1.DATA"),
        "SPE1.DATA",
    )
    ecl_config = Ecl100Config()
    run(ecl_config, ["SPE1.DATA", "--version=2019.3"])

    assert os.path.isfile("SPE1.DATA")


@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_failed_run(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1_ERROR.DATA"),
        "SPE1_ERROR.DATA",
    )
    ecl_config = Ecl100Config()
    eclrun_config = EclrunConfig(ecl_config, "2019.3")
    ecl_run = EclRun("SPE1_ERROR", None)
    with pytest.raises(Exception, match="ERROR"):
        ecl_run.runEclipse(eclrun_config=eclrun_config)


@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_failed_run_OK(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1_ERROR.DATA"),
        "SPE1_ERROR.DATA",
    )
    ecl_config = Ecl100Config()
    run(ecl_config, ["SPE1_ERROR", "--version=2019.3", "--ignore-errors"])


@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_no_hdf5_output_by_default_with_ecl100(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1.DATA"),
        "SPE1.DATA",
    )
    ecl_config = Ecl100Config()
    # check that by default .h5 file IS NOT produced
    run(ecl_config, ["SPE1.DATA", "--version=2019.3"])
    assert not os.path.exists("SPE1.h5")


@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_flag_to_produce_hdf5_output_with_ecl100(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1.DATA"),
        "SPE1.DATA",
    )
    ecl_config = Ecl100Config()
    # check that with flag .h5 file IS produced
    run(ecl_config, ["SPE1.DATA", "--version=2019.3", "--summary-conversion"])
    assert os.path.exists("SPE1.h5")


@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_mpi_run(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1_PARALLELL.DATA"),
        "SPE1_PARALLELL.DATA",
    )
    ecl_config = Ecl100Config()
    run(ecl_config, ["SPE1_PARALLELL.DATA", "--version=2019.3", "--num-cpu=2"])
    assert os.path.isfile("SPE1_PARALLELL.OUT")
    assert os.path.getsize("SPE1_PARALLELL.OUT") > 0


@pytest.mark.requires_eclipse
@pytest.mark.usefixtures("use_tmpdir", "init_eclrun_config")
def test_summary_block(source_root):
    shutil.copy(
        os.path.join(source_root, "test-data/eclipse/SPE1.DATA"),
        "SPE1.DATA",
    )
    ecl_config = Ecl100Config()
    ecl_run = EclRun("SPE1.DATA", None)
    ret_value = ecl_run.summary_block()
    assert ret_value is None

    ecl_run.runEclipse(eclrun_config=EclrunConfig(ecl_config, "2019.3"))
    ecl_sum = ecl_run.summary_block()
    assert isinstance(ecl_sum, EclSum)
