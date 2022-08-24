import json
import os
from textwrap import dedent

import pytest

from ert._c_wrappers.enkf import EnKFMain, ResConfig


@pytest.mark.usefixtures("use_tmpdir")
def test_transfer_var():
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

    run_context = ert.create_ensemble_experiment_run_context(iteration=0)
    ert.createRunPath(run_context)
    os.chdir("simulations/realization0")
    with open("jobs.json", "r") as f:
        data = json.load(f)
        env_data = data["global_environment"]
        assert env_data["FIRST"] == "TheFirstValue"
        assert env_data["SECOND"] == "TheSecondValue"

        path_data = data["global_update_path"]
        assert "TheThirdValue" == path_data["THIRD"]
        assert "TheFourthValue" == path_data["FOURTH"]
