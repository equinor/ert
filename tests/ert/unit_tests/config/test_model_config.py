from pathlib import Path
from unittest.mock import patch

import pytest

from ert.config import ConfigWarning, ModelConfig


def test_default_model_config_run_path():
    mc = ModelConfig(num_realizations=1)
    assert mc.runpath_format_string == "simulations/realization-<IENS>/iter-<ITER>"


def test_invalid_model_config_run_path():
    mc = ModelConfig(
        num_realizations=1, runpath_format_string="realization-no-specifier"
    )
    assert mc.runpath_format_string == "realization-no-specifier"


def test_suggested_deprecated_model_config_run_path():
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


@pytest.mark.parametrize(
    "total_space, used_space, to_warn, expected_warning",
    [
        pytest.param(
            10 * 1000**4,  # 10 TB
            9.75 * 1000**4,  # 9.75 TB
            False,
            None,
            id="Low disk space percentage on large disk",
        ),
        pytest.param(
            100 * 1000**3,  # 100 GB
            99 * 1000**3,  # 99 GB
            True,
            "Low disk space: 1.00 GB free on",
            id="Low disk space small disk",
        ),
        pytest.param(
            10 * 1000**5,  # 10 PB
            9.99994 * 1000**5,  # 9.99994 PB
            True,
            "Low disk space: 60.00 GB free on",
            id="Low disk space small disk",
        ),
        pytest.param(
            100 * 1000**3,  # 100 GB
            75 * 1000**3,  # 75 GB
            False,
            None,
            id="Sufficient disk space",
        ),
    ],
)
def test_that_when_the_runpath_disk_is_full_a_warning_is_given(
    tmp_path, recwarn, total_space, used_space, to_warn, expected_warning
):
    Path(tmp_path / "simulations").mkdir()
    runpath = f"{tmp_path!s}/simulations/realization-%d/iter-%d"
    with patch(
        "ert.config.model_config.shutil.disk_usage",
        return_value=(total_space, used_space, total_space - used_space),
    ):
        if to_warn:
            with pytest.warns(ConfigWarning, match=expected_warning):
                _ = ModelConfig(num_realizations=1, runpath_format_string=runpath)
        else:
            _ = ModelConfig(num_realizations=1, runpath_format_string=runpath)
            for w in recwarn:
                assert not issubclass(w.category, ConfigWarning)
