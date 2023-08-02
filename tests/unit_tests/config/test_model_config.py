from ert.config import ModelConfig


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
