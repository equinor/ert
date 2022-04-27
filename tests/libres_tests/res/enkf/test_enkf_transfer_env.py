import json
import os
from textwrap import dedent

from res.enkf import ResConfig, EnKFMain
from res.enkf.ert_run_context import ErtRunContext


def test_transfer_var(use_tmpdir):
    # Write a minimal config file with env
    with open("config_file.ert", "w") as fout:
        fout.write(
            dedent(
                """
        NUM_REALIZATIONS 1
        JOBNAME a_name_%d
        SETENV FIRST TheFirstValue
        SETENV SECOND TheSecondValue
        UPDATE_PATH   THIRD  TheThirdValue
        UPDATE_PATH   FOURTH TheFourthValue
        """
            )
        )
    res_config = ResConfig("config_file.ert")
    ert = EnKFMain(res_config)
    fs_manager = ert.getEnkfFsManager()

    model_config = ert.getModelConfig()
    run_context = ErtRunContext.ensemble_experiment(
        fs_manager.getCurrentFileSystem(),
        [True],
        model_config.getRunpathFormat(),
        model_config.getJobnameFormat(),
        ert.getDataKW(),
        0,
    )
    ert.getEnkfSimulationRunner().createRunPath(run_context)
    os.chdir("simulations/realization0")
    with open("jobs.json", "r") as f:
        data = json.load(f)
        env_data = data["global_environment"]
        assert env_data["FIRST"] == "TheFirstValue"
        assert env_data["SECOND"] == "TheSecondValue"

        path_data = data["global_update_path"]
        assert "TheThirdValue" == path_data["THIRD"]
        assert "TheFourthValue" == path_data["FOURTH"]
