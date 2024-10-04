import os.path
from pathlib import Path
from unittest.mock import patch

import pytest

from everest import util
from everest.bin.utils import report_on_previous_run
from everest.config import EverestConfig
from everest.config.everest_config import get_system_installed_jobs
from everest.config_keys import ConfigKeys
from everest.detached import ServerStatus
from everest.strings import SERVER_STATUS
from tests.everest.utils import (
    capture_streams,
    hide_opm,
    relpath,
    skipif_no_opm,
)

EGG_DATA = relpath(
    "../../test-data/everest/egg/eclipse/include/",
    "realizations/realization-0/eclipse/model/EGG.DATA",
)
SPE1_DATA = relpath("test_data/eclipse/SPE1.DATA")


@skipif_no_opm
def test_loadwells():
    wells = util.read_wellnames(SPE1_DATA)
    assert wells == ["PROD", "INJ"]


@skipif_no_opm
def test_loadgroups():
    groups = util.read_groupnames(EGG_DATA)
    assert {"FIELD", "PRODUC", "INJECT"} == set(groups)


@hide_opm
def test_loadwells_no_opm():
    with pytest.raises(RuntimeError):
        util.read_wellnames(SPE1_DATA)


@hide_opm
def test_loadgroups_no_opm():
    with pytest.raises(RuntimeError):
        util.read_groupnames(EGG_DATA)


def test_get_values(change_to_tmpdir):
    exp_dir = "the_config_directory"
    exp_file = "the_config_file"
    rel_out_dir = "the_output_directory"
    abs_out_dir = "/the_output_directory"
    os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, exp_file), "w", encoding="utf-8") as f:
        f.write(" ")

    config = EverestConfig.with_defaults(
        **{
            ConfigKeys.ENVIRONMENT: {
                ConfigKeys.OUTPUT_DIR: abs_out_dir,
                ConfigKeys.SIMULATION_FOLDER: "simulation_folder",
            },
            ConfigKeys.CONFIGPATH: Path(os.path.join(exp_dir, exp_file)),
        }
    )

    config.environment.output_folder = rel_out_dir


def test_makedirs(change_to_tmpdir):
    output_dir = os.path.join("unittest_everest_output")
    cwd = os.getcwd()

    # assert output dir (/tmp/tmpXXXX) is empty
    assert not os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 0

    # create output folder
    util.makedirs_if_needed(output_dir)

    # assert output folder created
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 1


def test_makedirs_already_exists(change_to_tmpdir):
    output_dir = os.path.join("unittest_everest_output")
    cwd = os.getcwd()

    # create outputfolder and verify it's existing
    util.makedirs_if_needed(output_dir)
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 1

    # run makedirs_if_needed again, verify nothing happened
    util.makedirs_if_needed(output_dir)
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 1


def test_makedirs_roll_existing(change_to_tmpdir):
    output_dir = os.path.join("unittest_everest_output")
    cwd = os.getcwd()

    # create outputfolder and verify it's existing
    util.makedirs_if_needed(output_dir)
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 1

    # run makedirs_if_needed again, verify old dir rolled
    util.makedirs_if_needed(output_dir, True)
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 2

    # run makedirs_if_needed again, verify old dir rolled
    util.makedirs_if_needed(output_dir, True)
    assert os.path.isdir(output_dir)
    assert len(os.listdir(cwd)) == 3


def test_get_everserver_status_path(copy_math_func_test_data_to_tmp):
    config = EverestConfig.load_file("config_minimal.yml")
    cwd = os.getcwd()
    session_path = os.path.join(
        cwd, "everest_output", "detached_node_output", ".session"
    )
    path = config.everserver_status_path
    expected_path = os.path.join(session_path, SERVER_STATUS)

    assert path == expected_path


def test_get_system_installed_job_names():
    job_names = get_system_installed_jobs()
    assert job_names is not None
    assert isinstance(job_names, list)
    assert len(job_names) > 0


@patch(
    "everest.bin.utils.everserver_status",
    return_value={"status": ServerStatus.failed, "message": "mock error"},
)
def test_report_on_previous_run(_, change_to_tmpdir):
    with open("config_file", "w", encoding="utf-8") as f:
        f.write(" ")
    config = EverestConfig.with_defaults(**{ConfigKeys.CONFIGPATH: "config_file"})
    with capture_streams() as (out, _):
        report_on_previous_run(config)
    lines = [line.strip() for line in out.getvalue().split("\n")]
    assert lines[0] == "Optimization run failed, with error: mock error"
