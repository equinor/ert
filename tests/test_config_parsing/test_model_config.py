from hypothesis import given

from ert._c_wrappers.enkf import ModelConfig, ResConfig

from .config_dict_generator import config_generators


def test_default_model_config_run_path(tmpdir):
    mc = ModelConfig(num_realizations=1)
    assert mc.runpath_format_string == "simulations/realization-<IENS>/iter-<ITER>"


def test_invalid_model_config_run_path(tmpdir):
    mc = ModelConfig(
        num_realizations=1, runpath_format_string="realization-no-specifier"
    )
    assert mc.runpath_format_string == "realization-no-specifier"


def test_deprecated_model_config_run_path(tmpdir):
    runpath = "simulations/realization-%d/iter-%d"
    mc = ModelConfig(num_realizations=1, runpath_format_string=runpath)
    assert mc.runpath_format_string == runpath


@given(config_generators())
def test_model_config_from_dict_and_user_config(tmp_path_factory, config_generator):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_dict:

        res_config_from_file = ResConfig(user_config_file=filename)
        res_config_from_dict = ResConfig(config_dict=config_dict)

        assert res_config_from_dict.model_config == res_config_from_file.model_config
