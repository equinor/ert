import os.path

from ert._c_wrappers.enkf import ModelConfig, ResConfig
from ert._c_wrappers.enkf.config_keys import ConfigKeys
from ert._c_wrappers.job_queue import ExtJoblist


def test_default_model_config_ens_path(tmpdir):
    with tmpdir.as_cwd():
        config_file = "test.ert"
        with open(config_file, "w") as f:
            f.write(
                """
NUM_REALIZATIONS  1
            """
            )
        res_config = ResConfig(config_file)
        # By default, the ensemble path is set to 'storage'
        default_ens_path = res_config.model_config.getEnspath()

        with open(config_file, "a") as f:
            f.write(
                """
ENSPATH storage
            """
            )

        # Set the ENSPATH in the config file
        res_config = ResConfig(config_file)
        set_in_file_ens_path = res_config.model_config.getEnspath()

        assert default_ens_path == set_in_file_ens_path

        config_dict = {ConfigKeys.NUM_REALIZATIONS: 1}
        dict_default_ens_path = ResConfig(
            config_dict=config_dict
        ).model_config.getEnspath()

        config_dict["ENSPATH"] = "storage"
        dict_set_ens_path = ResConfig(config_dict=config_dict).model_config.getEnspath()

        assert dict_default_ens_path == dict_set_ens_path
        assert dict_default_ens_path == default_ens_path


def test_default_model_config_run_path(tmpdir):
    assert ModelConfig(
        data_root=str(tmpdir),
        joblist=ExtJoblist(),
        refcase=None,
        config_dict={ConfigKeys.NUM_REALIZATIONS: 1},
    ).getRunpathFormat()._str() == os.path.abspath(
        "simulations/realization-<IENS>/iter-<ITER>"
    )
