import os

import pytest

from ert._c_wrappers.config import ConfigValidationError
from ert._c_wrappers.enkf import ConfigKeys, ResConfig


@pytest.mark.usefixtures("copy_minimum_case")
def test_eclbase_and_jobname():
    res_config = ResConfig(
        config={
            "INTERNALS": {
                "CONFIG_DIRECTORY": ".",
            },
            "SIMULATION": {
                "QUEUE_SYSTEM": {
                    "JOBNAME": "JOBNAME%d",
                },
                "RUNPATH": "/tmp/simulations/realization-<IENS>/iter-<ITER>",
                "NUM_REALIZATIONS": 1,
                "JOB_SCRIPT": "script.sh",
                "ENSPATH": "Ensemble",
                "ECLBASE": "ECLBASE%d",
            },
        }
    )
    assert res_config.model_config.jobname_format_string == "JOBNAME%d"


@pytest.mark.usefixtures("copy_minimum_case")
def test_eclbase():
    res_config = ResConfig(
        config={
            "INTERNALS": {
                "CONFIG_DIRECTORY": ".",
            },
            "SIMULATION": {
                "RUNPATH": "/tmp/simulations/realization-<IENS>/iter-<ITER>",
                "NUM_REALIZATIONS": 1,
                "JOB_SCRIPT": "script.sh",
                "ENSPATH": "Ensemble",
                "ECLBASE": "ECLBASE%d",
            },
        }
    )

    assert res_config.model_config.jobname_format_string == "ECLBASE%d"


def test_that_summary_given_without_eclbase_gives_error_from_file(tmp_path):
    (tmp_path / "config.ert").write_text("NUM_REALIZATIONS 1\nSUMMARY summary")
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="When using SUMMARY keyword, the config must also specify ECLBASE",
    ):
        ResConfig(user_config_file=str(tmp_path / "config.ert"))


def test_that_summary_given_without_eclbase_gives_error_from_dict(tmp_path):
    config_dict = {
        "NUM_REALIZATIONS": "1",
        "ENSPATH": os.path.join(tmp_path, "storage"),
        "SUMMARY": "summary",
    }
    with pytest.raises(
        expected_exception=ConfigValidationError,
        match="When using SUMMARY keyword, the config must also specify ECLBASE",
    ):
        ResConfig(config_dict=config_dict)


@pytest.mark.usefixtures("copy_minimum_case")
def test_jobname():
    res_config = ResConfig(
        config={
            "INTERNALS": {
                "CONFIG_DIRECTORY": ".",
            },
            "SIMULATION": {
                "QUEUE_SYSTEM": {
                    "JOBNAME": "JOBNAME%d",
                },
                "RUNPATH": "/tmp/simulations/realization-<IENS>/iter-<ITER>",
                "NUM_REALIZATIONS": 1,
                "JOB_SCRIPT": "script.sh",
                "ENSPATH": "Ensemble",
            },
        }
    )
    assert res_config.model_config.jobname_format_string == "JOBNAME%d"


def test_model_config_dict_constructor(setup_case):
    res_config_from_file = setup_case("configuration_tests", "model_config.ert")
    config_dict = {
        ConfigKeys.JOBNAME: "model_config_test",
        ConfigKeys.RUNPATH: "/tmp/simulations/realization-<IENS>/iter-<ITER>",
        ConfigKeys.NUM_REALIZATIONS: 10,
        ConfigKeys.ENSPATH: os.path.join(os.getcwd(), "Ensemble"),
        ConfigKeys.TIME_MAP: os.path.join(os.getcwd(), "input/refcase/time_map.txt"),
        ConfigKeys.OBS_CONFIG: os.path.join(
            os.getcwd(), "input/observations/observations.txt"
        ),
        ConfigKeys.DATAROOT: os.getcwd(),
        ConfigKeys.REFCASE: "input/refcase/SNAKE_OIL_FIELD",
        ConfigKeys.HISTORY_SOURCE: "REFCASE_HISTORY",
        ConfigKeys.GEN_KW_EXPORT_NAME: "parameter_test.json",
        ConfigKeys.FORWARD_MODEL: [
            ["COPY_FILE", "<FROM>=input/schedule.sch,<TO>=output/schedule_copy.sch"],
            [
                "SNAKE_OIL_SIMULATOR",
                "",
            ],
            [
                "SNAKE_OIL_NPV",
                "",
            ],
            [
                "SNAKE_OIL_DIFF",
                "",
            ],
        ],
        # needed to make up for lack of site config handling on config dict path
        ConfigKeys.INSTALL_JOB: [
            ("SNAKE_OIL_SIMULATOR", "input/jobs/SNAKE_OIL_SIMULATOR"),
            ("SNAKE_OIL_NPV", "input/jobs/SNAKE_OIL_NPV"),
            ("SNAKE_OIL_DIFF", "input/jobs/SNAKE_OIL_DIFF"),
        ],
        # unelegant replication of the path resolution of site config
        ConfigKeys.INSTALL_JOB_DIRECTORY: [
            os.path.normpath(
                os.path.realpath(
                    os.path.join(
                        os.path.dirname(__file__),
                        (
                            "../../../../../"
                            "src/ert/shared/share/ert/forward-models/old_style/"
                        ),
                    )
                )
            )
        ],
    }
    res_config_from_dict = ResConfig(config_dict=config_dict)
    assert res_config_from_dict.model_config == res_config_from_file.model_config
