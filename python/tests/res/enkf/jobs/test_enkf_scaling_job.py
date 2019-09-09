import pytest
import os
import shutil
import random
import yaml
import configsuite
import numpy as np

from copy import deepcopy

from res.enkf import ResConfig, EnKFMain, ConfigKeys
from res.enkf.jobs import enkf_scaling_job, enkf_scaling_job_config
from ecl.util.util import BoolVector
from res.enkf import ErtRunContext

_TEST_DATA_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), os.path.join("..", "..","..","..","test-data")
))


def p(x):
    return 0.5 * x ** 2 + x + 3

def assert_obs_vector(vector, val_1, index_list=None, val_2=None):
    if index_list is None:
        index_list = []
    for node in vector:
        for index in range(len(node)):
            if index in index_list:
                assert node.getStdScaling(index) == val_2
            else:
                assert node.getStdScaling(index) == val_1


@pytest.fixture(scope="module")
def setup_ert(tmp_path_factory):

    config = {
        ConfigKeys.SIMULATION:
            {
            ConfigKeys.QUEUE_SYSTEM:
                {
                    ConfigKeys.JOBNAME: "poly%d",
                    ConfigKeys.QUEUE_SYSTEM: "LOCAL",
                    ConfigKeys.MAX_SUBMIT: 50,
                },
            ConfigKeys.RUNPATH: "poly_out/real_%d/iter_%d",
            ConfigKeys.NUM_REALIZATIONS: 5,
            ConfigKeys.MIN_REALIZATIONS: 1,
            ConfigKeys.OBS_CONFIG: "observations",
            ConfigKeys.TIME_MAP: "time_map",
            ConfigKeys.SEED:
                {ConfigKeys.RANDOM_SEED: 123},
            ConfigKeys.GEN_DATA:
                [
                    {
                        ConfigKeys.NAME: "POLY_RES",
                        "RESULT_FILE": "poly_%d.out",
                        "REPORT_STEPS": 0,
                    }
                ],
            ConfigKeys.GEN_KW:
                [
                    {
                        ConfigKeys.NAME: "COEFFS",
                        ConfigKeys.TEMPLATE: "coeff.tmpl",
                        ConfigKeys.OUT_FILE: "coeffs.json",
                        ConfigKeys.PARAMETER_FILE: "coeff_priors"
                    }
                ],
                ConfigKeys.INSTALL_JOB:
                [
                    {
                        ConfigKeys.NAME: "poly_eval",
                        ConfigKeys.PATH: "POLY_EVAL"
                    },
                ],
            ConfigKeys.SIMULATION_JOB:
                [
                    {ConfigKeys.NAME: "poly_eval"},
                ],
            }
        }

    temp_path = tmp_path_factory.mktemp("poly_case")

    cwd = os.getcwd()
    os.chdir(temp_path.as_posix())

    test_files = ["time_map", "POLY_EVAL", "poly_eval.py", "coeff.tmpl", "coeff_priors"]
    for test_file in test_files:
        shutil.copy(os.path.join(_TEST_DATA_DIR, "local", "poly_normal", test_file), temp_path.as_posix())

    random.seed(123)
    observations = [(p(x) + random.gauss(0, 0.25*x**2 + 0.1), 0.25*x**2 + 0.1) for x in range(10)]
    with open("poly_obs_data.txt", "w") as fout:
        for value, error in observations:
            fout.write("{:.1f} {:.1f}\n".format(value, error))

    obs_config = ("GENERAL_OBSERVATION POLY_OBS {"
                  "\n   DATA       = POLY_RES;"
                  "\n   RESTART    = 0;"
                  "\n   OBS_FILE   = poly_obs_data.txt;\n};\n"
                  )

    with open("observations", "w") as fout:
           fout.write(obs_config)

    res_config = ResConfig(config=config)

    ert = EnKFMain(res_config)

    sim_fs = ert.getEnkfFsManager().getCurrentFileSystem()

    ensamble_mask = BoolVector(default_value=True, initial_size=ert.getEnsembleSize())
    model_config = ert.getModelConfig()
    runpath_format = model_config.getRunpathFormat()
    jobname_format = model_config.getJobnameFormat()

    subst_list = ert.getDataKW()
    run_context = ErtRunContext.ensemble_experiment(
        sim_fs, ensamble_mask, runpath_format, jobname_format, subst_list, 0
    )

    ert.createRunpath(run_context)
    ert.getEnkfSimulationRunner().runEnsembleExperiment(
        ert.get_queue_config().create_job_queue(), run_context)

    yield config

    os.chdir(cwd)


def test_old_enkf_scaling_job(setup_ert):

    res_config = ResConfig(config=setup_ert)
    ert = EnKFMain(res_config)

    obs = ert.getObservations()
    obs_vector = obs["POLY_OBS"]

    assert_obs_vector(obs_vector, 1.0)

    job = ert.getWorkflowList().getJob("STD_SCALE_CORRELATED_OBS")
    job.run(ert, ["POLY_OBS"])

    assert_obs_vector(obs_vector, 3.1622776601683795)