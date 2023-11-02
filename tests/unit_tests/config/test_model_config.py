import pytest

from ert.config import ModelConfig
from ert.config.parsing import ConfigKeys, ConfigValidationError


def test_default_model_config_run_path(tmpdir):
    mc = ModelConfig(num_realizations=1)
    assert mc.runpath_format_string == "simulations/realization-<IENS>/iter-<ITER>"


def test_invalid_model_config_run_path(tmpdir):
    mc = ModelConfig(
        num_realizations=1, runpath_format_string="realization-no-specifier"
    )
    assert mc.runpath_format_string == "realization-no-specifier"


def test_suggested_deprecated_model_config_run_path(tmpdir):
    runpath = "simulations/realization-%d/iter-%d"
    suggested_path = "simulations/realization-<IENS>/iter-<ITER>"
    mc = ModelConfig(num_realizations=1, runpath_format_string=runpath)
    assert mc.runpath_format_string == suggested_path


@pytest.mark.filterwarnings("ignore::ert.config.ConfigWarning")
@pytest.mark.parametrize(
    "parameters, expected",
    [
        pytest.param(
            {
                "eclbase_format_string": "ECLBASE%d",
                "jobname_format_string": "JOBNAME%d",
            },
            "JOBNAME<IENS>",
            id="ECLBASE does not overwrite JOBNAME",
        ),
        pytest.param(
            {"eclbase_format_string": "ECLBASE%d"},
            "ECLBASE<IENS>",
            id="ECLBASE is also used as JOBNAME",
        ),
        pytest.param(
            {
                "eclbase_format_string": "ECLBASE%d",
                "jobname_format_string": "JOBNAME%d",
            },
            "JOBNAME<IENS>",
            id="JOBNAME is used when no ECLBASE is present",
        ),
        pytest.param(
            {},
            "<CONFIG_FILE>-<IENS>",
            id="JOBNAME defaults to <CONFIG_FILE>-<IENS>",
        ),
    ],
)
def test_model_config_jobname_and_eclbase(parameters, expected):
    assert ModelConfig(**parameters).jobname_format_string == expected


def test_that_invalid_time_map_file_raises_config_validation_error(tmpdir):
    with tmpdir.as_cwd():
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("invalid")

        with pytest.raises(ConfigValidationError, match="Could not read timemap file"):
            _ = ModelConfig(time_map_file="time_map.txt")
