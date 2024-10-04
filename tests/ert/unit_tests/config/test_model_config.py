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
    "extra_config, expected",
    [
        pytest.param(
            {"ECLBASE": "ECLBASE%d", "JOBNAME": "JOBNAME%d"},
            "JOBNAME<IENS>",
            id="ECLBASE does not overwrite JOBNAME",
        ),
        pytest.param(
            {"ECLBASE": "ECLBASE%d"},
            "ECLBASE<IENS>",
            id="ECLBASE is also used as JOBNAME",
        ),
        pytest.param(
            {"ECLBASE": "ECLBASE%d", "JOBNAME": "JOBNAME%d"},
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
def test_model_config_jobname_and_eclbase(extra_config, expected):
    config_dict = {"NUM_REALIZATIONS": 1, "ENSPATH": "Ensemble", **extra_config}
    assert ModelConfig.from_dict(config_dict).jobname_format_string == expected


def test_that_invalid_time_map_file_raises_config_validation_error(tmpdir):
    with tmpdir.as_cwd():
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("invalid")

        with pytest.raises(ConfigValidationError, match="Could not read timemap file"):
            _ = ModelConfig.from_dict({ConfigKeys.TIME_MAP: "time_map.txt"})
