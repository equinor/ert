import os.path

from hypothesis import given

from ert._c_wrappers.enkf import ModelConfig, ResConfig
from ert._c_wrappers.enkf.config_keys import ConfigKeys

from .config_dict_generator import config_generators


def test_default_model_config_ens_path(tmpdir):
    with tmpdir.as_cwd():
        config_file = "test.ert"
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(
                """
NUM_REALIZATIONS  1
            """
            )
        res_config = ResConfig(config_file)
        # By default, the ensemble path is set to 'storage'
        default_ens_path = res_config.model_config.ens_path

        with open(config_file, "a", encoding="utf-8") as f:
            f.write(
                """
ENSPATH storage
            """
            )

        # Set the ENSPATH in the config file
        res_config = ResConfig(config_file)
        set_in_file_ens_path = res_config.model_config.ens_path

        assert default_ens_path == set_in_file_ens_path

        config_dict = {
            ConfigKeys.NUM_REALIZATIONS: 1,
            "ENSPATH": os.path.join(os.getcwd(), "storage"),
        }

        dict_set_ens_path = ResConfig(config_dict=config_dict).model_config.ens_path

        assert dict_set_ens_path == config_dict["ENSPATH"]


def test_default_model_config_run_path(tmpdir):
    assert (
        ModelConfig(
            num_realizations=1,
        ).runpath_format_string
        == "simulations/realization-%d/iter-%d"
    )


@given(config_generators())
def test_model_config_from_dict_and_user_config(tmp_path_factory, config_generator):
    filename = "config.ert"
    with config_generator(tmp_path_factory, filename) as config_dict:

        res_config_from_file = ResConfig(user_config_file=filename)
        res_config_from_dict = ResConfig(config_dict=config_dict)

        assert res_config_from_dict.model_config == res_config_from_file.model_config
