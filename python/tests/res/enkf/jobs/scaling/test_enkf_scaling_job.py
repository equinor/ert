import pytest
import os
import shutil
import random
import configsuite
import yaml

import numpy as np
import pandas as pd

from copy import deepcopy
from collections import namedtuple

from res.enkf import ResConfig, EnKFMain, ConfigKeys
from res.enkf.jobs.scaling import scaling_job, job_config, measured_data, scaled_matrix
from ecl.util.util import BoolVector
from res.enkf import ErtRunContext

_TEST_DATA_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        os.path.join("..", "..", "..", "..", "..", "test-data"),
    )
)


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


@pytest.fixture()
def setup_tmpdir(tmpdir):
    cwd = os.getcwd()
    tmpdir.chdir()
    yield
    os.chdir(cwd)


@pytest.fixture()
def valid_poly_config():

    valid_config_data = {
        "CALCULATE_KEYS": {"keys": [{"key": "POLY_OBS"}]},
        "UPDATE_KEYS": {"keys": [{"key": "POLY_OBS"}]},
    }

    schema = job_config.build_schema()
    config = configsuite.ConfigSuite(valid_config_data, schema)
    yield config


@pytest.fixture(scope="module")
def setup_ert(tmp_path_factory):

    config = {
        ConfigKeys.SIMULATION: {
            ConfigKeys.QUEUE_SYSTEM: {
                ConfigKeys.JOBNAME: "poly_%d",
                ConfigKeys.QUEUE_SYSTEM: "LOCAL",
                ConfigKeys.MAX_SUBMIT: 50,
            },
            ConfigKeys.RUNPATH: "poly_out/real_%d/iter_%d",
            ConfigKeys.NUM_REALIZATIONS: 5,
            ConfigKeys.MIN_REALIZATIONS: 1,
            ConfigKeys.OBS_CONFIG: "observations",
            ConfigKeys.TIME_MAP: "time_map",
            ConfigKeys.SEED: {ConfigKeys.RANDOM_SEED: 123},
            ConfigKeys.GEN_DATA: [
                {
                    ConfigKeys.NAME: "POLY_RES",
                    "RESULT_FILE": "poly_%d.out",
                    "REPORT_STEPS": 0,
                }
            ],
            ConfigKeys.GEN_KW: [
                {
                    ConfigKeys.NAME: "COEFFS",
                    ConfigKeys.TEMPLATE: "coeff.tmpl",
                    ConfigKeys.OUT_FILE: "coeffs.json",
                    ConfigKeys.PARAMETER_FILE: "coeff_priors",
                }
            ],
            ConfigKeys.INSTALL_JOB: [
                {ConfigKeys.NAME: "poly_eval", ConfigKeys.PATH: "POLY_EVAL"}
            ],
            ConfigKeys.SIMULATION_JOB: [{ConfigKeys.NAME: "poly_eval"}],
        }
    }

    temp_path = tmp_path_factory.mktemp("poly_case")

    cwd = os.getcwd()
    os.chdir(temp_path.as_posix())

    test_files = ["time_map", "POLY_EVAL", "poly_eval.py", "coeff.tmpl", "coeff_priors"]
    for test_file in test_files:
        shutil.copy(
            os.path.join(_TEST_DATA_DIR, "local", "poly_normal", test_file),
            temp_path.as_posix(),
        )

    random.seed(123)
    observations = [
        (p(x) + random.gauss(0, 0.25 * x ** 2 + 0.1), 0.25 * x ** 2 + 0.1)
        for x in range(10)
    ]
    with open("poly_obs_data.txt", "w") as fout:
        for value, error in observations:
            fout.write("{:.1f} {:.1f}\n".format(value, error))

    obs_config = (
        "GENERAL_OBSERVATION POLY_OBS {"
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
        ert.get_queue_config().create_job_queue(), run_context
    )

    yield config

    os.chdir(cwd)


def get_config(index_list_calc=None, index_list_update=None):
    schema = job_config.build_schema()
    default_values = job_config.get_default_values()
    config = {
        "UPDATE_KEYS": {"keys": [{"key": "POLY_OBS"}]},
        "CALCULATE_KEYS": {"keys": [{"key": "POLY_OBS"}]},
    }

    if index_list_update:
        config["UPDATE_KEYS"]["keys"][0].update({"index": index_list_update})
    if index_list_calc:
        config["CALCULATE_KEYS"]["keys"][0].update({"index": index_list_calc})

    return configsuite.ConfigSuite(config, schema, layers=(default_values,)).snapshot


def test_old_enkf_scaling_job(setup_ert):

    res_config = ResConfig(config=setup_ert)
    ert = EnKFMain(res_config)

    obs = ert.getObservations()
    obs_vector = obs["POLY_OBS"]

    assert_obs_vector(obs_vector, 1.0)

    job = ert.getWorkflowList().getJob("STD_SCALE_CORRELATED_OBS")
    job.run(ert, ["POLY_OBS"])

    assert_obs_vector(obs_vector, 3.1622776601683795)


def test_python_version_of_enkf_scaling_job(setup_ert):
    res_config = ResConfig(config=setup_ert)
    ert = EnKFMain(res_config)

    obs = ert.getObservations()
    obs_vector = obs["POLY_OBS"]

    assert_obs_vector(obs_vector, 1.0)

    job_config = {"CALCULATE_KEYS": {"keys": [{"key": "POLY_OBS"}]}}

    with open("job_config.yml", "w") as fout:
        yaml.dump(job_config, fout)

    job = ert.getWorkflowList().getJob("CORRELATE_OBSERVATIONS")
    job.run(ert, ["job_config.yml"])

    assert_obs_vector(obs_vector, np.sqrt(10.0))

    job_config["CALCULATE_KEYS"]["keys"][0].update({"index": [1, 2, 3]})
    with open("job_config.yml", "w") as fout:
        yaml.dump(job_config, fout)

    job.run(ert, ["job_config.yml"])

    assert_obs_vector(
        obs_vector,
        np.sqrt(10.0),
        index_list=job_config["CALCULATE_KEYS"]["keys"][0]["index"],
        val_2=np.sqrt(3.0),
    )


def test_compare_different_jobs(setup_ert):

    res_config = ResConfig(config=setup_ert)
    ert = EnKFMain(res_config)

    obs = ert.getObservations()
    obs_vector = obs["POLY_OBS"]

    assert_obs_vector(obs_vector, 1.0)

    job = ert.getWorkflowList().getJob("STD_SCALE_CORRELATED_OBS")
    job.run(ert, ["POLY_OBS"])

    # Result of old job:
    assert_obs_vector(obs_vector, 3.1622776601683795)

    scaling_job._observation_scaling(ert, get_config())

    # Result of new job with no sub-indexing:
    assert_obs_vector(obs_vector, 3.1622776601683795)


def test_enkf_scale_job(setup_ert):

    res_config = ResConfig(config=setup_ert)
    ert = EnKFMain(res_config)
    obs = ert.getObservations()

    obs_vector = obs["POLY_OBS"]

    assert_obs_vector(obs_vector, 1.0)

    index_list_update = [0, 1, 2, 3, 4]
    scaling_job._observation_scaling(
        ert, get_config(index_list_update=index_list_update)
    )

    assert_obs_vector(obs_vector, 1.0, index_list_update, 3.1622776601683795)

    scaling_job._observation_scaling(ert, get_config())

    assert_obs_vector(obs_vector, 3.1622776601683795)

    index_list_calc = [0, 2, 3, 4]
    scaling_job._observation_scaling(ert, get_config(index_list_calc=index_list_calc))

    assert_obs_vector(obs_vector, 2.0)

    index_list_update = [0, 2, 3, 4]
    index_list_calc = [0, 1]
    scaling_job._observation_scaling(
        ert,
        get_config(
            index_list_calc=index_list_calc, index_list_update=index_list_update
        ),
    )

    assert_obs_vector(obs_vector, 2.0, index_list_update, 1.4142135623730951)


def test_create_observation_vectors(setup_ert, valid_poly_config):

    res_config = ResConfig(config=setup_ert)
    ert = EnKFMain(res_config)
    obs = ert.getObservations()

    new_events = scaling_job._create_active_lists(
        obs, valid_poly_config.snapshot.UPDATE_KEYS.keys
    )

    keys = [event.key for event in new_events]

    assert "POLY_OBS" in keys
    assert "POLY_RES" not in keys


def test_is_subset():
    example_list = ["a", "b", "c"]

    assert scaling_job.is_subset(example_list, ["a"]) == []
    assert scaling_job.is_subset(example_list, ["a", "c"]) == []
    assert scaling_job.is_subset(example_list, ["a", "b", "c"]) == []
    assert len(scaling_job.is_subset(example_list, ["d"])) == 1


def test_has_keys(setup_ert):

    res_config = ResConfig(config=setup_ert)
    ert = EnKFMain(res_config)
    obs = ert.getObservations()

    valid_key = ["POLY_OBS"]
    assert scaling_job.has_keys(obs, valid_key) == []

    invalid_key = ["NOT_A_KEY"]
    assert len(scaling_job.has_keys(obs, invalid_key)) == 1

    valid_and_invalid_keys = ["POLY_OBS", "NOT_A_KEY"]
    assert len(scaling_job.has_keys(obs, valid_and_invalid_keys)) == 1

    invalid_keys = ["POLY_NOT", "NOT_A_KEY"]
    assert len(scaling_job.has_keys(obs, invalid_keys)) == 2


def test_valid_job(setup_ert, valid_poly_config):
    res_config = ResConfig(config=setup_ert)
    ert = EnKFMain(res_config)
    obs = ert.getObservations()

    assert scaling_job.valid_job(
        obs,
        valid_poly_config,
        ert.getEnsembleSize(),
        ert.getEnkfFsManager().getCurrentFileSystem(),
    )


def test_to_int_list():
    expected_result = list(range(6))

    valid_inputs = [
        "0,1,2-5",
        [0, 1, "2-5"],
        [0, 1, 2, 3, 4, 5],
        [0, 1, "2-3", "4-5"],
        "0-5",
        "0-1,2,3-5",
        ["0,1,2-5"],
    ]

    for valid_input in valid_inputs:
        assert job_config._to_int_list(valid_input) == expected_result

    expected_result = [1]

    assert job_config._to_int_list([1]) == expected_result


def test_min_value():
    assert not job_config._min_value(-1)
    assert job_config._min_value(0)


def test_expand_input():

    expected_result = {
        "UPDATE_KEYS": {"keys": [{"key": "key_4"}, {"key": "key_5"}, {"key": "key_6"}]},
        "CALCULATE_KEYS": {
            "keys": [{"key": "key_1"}, {"key": "key_2"}, {"key": "key_3"}]
        },
    }

    valid_config = deepcopy(expected_result)

    assert job_config._expand_input(deepcopy(expected_result)) == expected_result

    copy_of_valid_config = deepcopy(valid_config)
    copy_of_valid_config.pop("UPDATE_KEYS")

    expected_result = {
        "UPDATE_KEYS": {"keys": [{"key": "key_1"}, {"key": "key_2"}, {"key": "key_3"}]},
        "CALCULATE_KEYS": {
            "keys": [{"key": "key_1"}, {"key": "key_2"}, {"key": "key_3"}]
        },
    }

    assert job_config._expand_input(copy_of_valid_config) == expected_result


def test_config_setup():

    valid_config_data = {
        "CALCULATE_KEYS": {"keys": [{"key": "first_key"}, {"key": "second_key"}]}
    }

    schema = job_config.build_schema()
    config = configsuite.ConfigSuite(valid_config_data, schema)
    assert config.valid

    valid_config_data = {
        "CALCULATE_KEYS": {"keys": [{"key": "first_key"}, {"key": "second_key"}]},
        "UPDATE_KEYS": {"keys": [{"index": [1, 2, 3], "key": "first_key"}]},
    }

    schema = job_config.build_schema()
    config = configsuite.ConfigSuite(valid_config_data, schema)
    assert config.valid

    invalid_too_short_index_list = {
        "UPDATE_KEYS": {"keys": [{"index": "1", "key": ["a_key"]}]}
    }

    config = configsuite.ConfigSuite(invalid_too_short_index_list, schema)
    assert not config.valid

    invalid_missing_required_keyword = {
        "CALCULATE_KEYS": {"keys": [{"key": "a_key"}]},
        "UPDATE_KEYS": {"index": "1-5"},
    }

    config = configsuite.ConfigSuite(invalid_missing_required_keyword, schema)
    assert not config.valid

    invalid_negative_index = {
        "CALCULATE_KEYS": {"keys": [{"key": "first_key"}, {"key": "second_key"}]},
        "UPDATE_KEYS": {"keys": [{"index": [-1, 2, 3], "key": "first_key"}]},
    }

    schema = job_config.build_schema()
    config = configsuite.ConfigSuite(invalid_negative_index, schema)
    assert not config.valid


def test_main_entry_point(setup_ert):

    res_config = ResConfig(config=setup_ert)
    ert = EnKFMain(res_config)

    arguments = {
        "CALCULATE_KEYS": {"keys": [{"key": "POLY_OBS"}]},
        "UPDATE_KEYS": {"keys": [{"key": "POLY_OBS", "index": [1, 2, 3]}]},
    }

    scaling_job.scaling_job(ert, arguments)

    obs = ert.getObservations()
    obs_vector = obs["POLY_OBS"]

    assert_obs_vector(
        obs_vector,
        1.0,
        arguments["UPDATE_KEYS"]["keys"][0]["index"],
        3.1622776601683795,
    )

    arguments["CALCULATE_KEYS"]["keys"][0].update({"index": [7, 8, 9]})
    scaling_job.scaling_job(ert, arguments)
    assert_obs_vector(
        obs_vector,
        1.0,
        arguments["UPDATE_KEYS"]["keys"][0]["index"],
        1.7320508075688772,
    )


def test_get_scaling_factor():
    new_event = namedtuple("named_dict", ["keys", "threshold"])
    event = new_event(["one_random_key"], 0.95)
    np.random.seed(123)
    input_matrix = np.random.rand(10, 10)

    matrix = scaled_matrix.DataMatrix(pd.DataFrame(data=input_matrix))

    assert matrix.get_scaling_factor(event) == np.sqrt(10 / 4.0)


def test_get_nr_primary_components():
    np.random.seed(123)
    input_matrix = np.random.rand(10, 10)

    matrix = scaled_matrix.DataMatrix

    assert matrix._get_nr_primary_components(input_matrix, 0.0) == 1
    assert matrix._get_nr_primary_components(input_matrix, 0.83) == 2
    assert matrix._get_nr_primary_components(input_matrix, 0.90) == 3
    assert matrix._get_nr_primary_components(input_matrix, 0.95) == 4
    assert matrix._get_nr_primary_components(input_matrix, 0.99) == 6


def test_std_normalization():
    input_matrix = pd.DataFrame(np.ones((3, 3)))
    input_matrix.loc["OBS"] = np.ones(3)
    input_matrix.loc["STD"] = np.ones(3) * 0.1
    expected_matrix = [[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
    matrix = scaled_matrix.DataMatrix(pd.concat({"A_KEY": input_matrix}, axis=1))
    result = matrix.std_normalization(["A_KEY"])
    assert (result.loc[[0, 1, 2]].values == expected_matrix).all()


def test_filter_on_column_index(setup_ert):
    res_config = ResConfig(config=setup_ert)
    ert = EnKFMain(res_config)

    data_matrix = measured_data.MeasuredData(ert, ["POLY_OBS"])

    matrix = np.random.rand(10, 10)

    index_lists = [[0, 1], [1, 2, 3], [1, 2, 3, 4, 5]]
    for index_list in index_lists:
        data_matrix.data = pd.DataFrame(matrix)
        data_matrix.data = data_matrix.filter_on_column_index(
            data_matrix.data, index_list
        )
        assert data_matrix.data.shape == (10, len(index_list))

    data_matrix.data = pd.DataFrame(matrix)
    with pytest.raises(IndexError):
        data_matrix._filter_on_column_index(data_matrix.data, [11])


@pytest.mark.usefixtures("setup_tmpdir")
def test_add_wildcards():

    test_data_dir = os.path.join(_TEST_DATA_DIR, "local", "snake_oil_field")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")
    ert = EnKFMain(res_config)

    expected_dict = {
        "ANOTHER_KEY": "something",
        "CALCULATE_KEYS": {
            "keys": [
                {"key": "WOPR_OP1_108"},
                {"key": "WOPR_OP1_144"},
                {"key": "WOPR_OP1_190"},
                {"key": "WOPR_OP1_9"},
                {"key": "WOPR_OP1_36"},
                {"key": "WOPR_OP1_72"},
                {"key": "FOPR"},
            ]
        },
        "UPDATE_KEYS": {
            "keys": [
                {"key": "WOPR_OP1_108"},
                {"key": "WOPR_OP1_144"},
                {"key": "WOPR_OP1_190"},
                {"key": "FOPR"},
            ]
        },
    }

    user_config = {
        "ANOTHER_KEY": "something",
        "CALCULATE_KEYS": {"keys": [{"key": "WOPR_*"}, {"key": "FOPR"}]},
        "UPDATE_KEYS": {"keys": [{"key": "WOPR_OP1_1*"}, {"key": "FOPR"}]},
    }

    result_dict = scaling_job._find_and_expand_wildcards(
        ert.getObservations().getMatchingKeys, user_config
    )

    assert result_dict == expected_dict


@pytest.mark.usefixtures("setup_tmpdir")
def test_add_observation_vectors():

    valid_config_data = {"UPDATE_KEYS": {"keys": [{"key": "WOPR_OP1_108"}]}}

    schema = job_config.build_schema()
    config = configsuite.ConfigSuite(valid_config_data, schema)

    test_data_dir = os.path.join(_TEST_DATA_DIR, "local", "snake_oil_field")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")

    ert = EnKFMain(res_config)

    obs = ert.getObservations()

    new_events = scaling_job._create_active_lists(obs, config.snapshot.UPDATE_KEYS.keys)

    keys = [event.key for event in new_events]

    assert "WOPR_OP1_108" in keys
    assert "WOPR_OP1_144" not in keys


@pytest.mark.usefixtures("setup_tmpdir")
def test_main_entry_point_summary_data_calc():

    arguments = {
        "CALCULATE_KEYS": {"keys": [{"key": "WOPR_OP1_108"}, {"key": "WOPR_OP1_144"}]}
    }

    test_data_dir = os.path.join(_TEST_DATA_DIR, "local", "snake_oil")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("snake_oil.ert")

    ert = EnKFMain(res_config)

    obs = ert.getObservations()

    obs_vector = obs["WOPR_OP1_108"]

    for index, node in enumerate(obs_vector):
        assert node.getStdScaling(index) == 1.0

    scaling_job.scaling_job(ert, arguments)

    for index, node in enumerate(obs_vector):
        assert node.getStdScaling(index) == np.sqrt((2.0 * 6.0) / 2.0)

    arguments["CALCULATE_KEYS"]["keys"][0].update({"index": [1, 2, 3]})
    arguments["CALCULATE_KEYS"]["keys"][1].update({"index": [1, 2, 3]})

    with pytest.raises(ValueError):  # Will give an empty data set
        scaling_job.scaling_job(ert, arguments)

    arguments["CALCULATE_KEYS"]["keys"][0].update({"index": [8, 35, 71]})
    arguments["CALCULATE_KEYS"]["keys"][1].update({"index": [8, 35, 71]})
    scaling_job.scaling_job(ert, arguments)

    for index, node in enumerate(obs_vector):
        if index in arguments["CALCULATE_KEYS"]["keys"][0]["index"]:
            assert node.getStdScaling(index) == np.sqrt((2.0 * 6.0) / 1.0)
        else:
            assert node.getStdScaling(index) == np.sqrt((2.0 * 6.0) / 2.0)


@pytest.mark.equinor_test
@pytest.mark.usefixtures("setup_tmpdir")
def test_main_entry_point_summary_data_update():
    arguments = {
        "CALCULATE_KEYS": {"keys": [{"key": "WWCT:OP_1"}, {"key": "WWCT:OP_2"}]},
        "UPDATE_KEYS": {"keys": [{"key": "WWCT:OP_2", "index": [1, 2, 3, 4, 5]}]},
    }

    test_data_dir = os.path.join(_TEST_DATA_DIR, "Equinor", "config", "obs_testing")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("config")
    ert = EnKFMain(res_config)

    obs = ert.getObservations()
    obs_vector = obs["WWCT:OP_2"]

    scaling_job.scaling_job(ert, arguments)

    for index, node in enumerate(obs_vector):
        if index in arguments["UPDATE_KEYS"]["keys"][0]["index"]:
            assert node.getStdScaling(index) == np.sqrt(61.0 * 2.0)
        else:
            assert node.getStdScaling(index) == 1.0

    obs_vector = obs["WWCT:OP_1"]

    scaling_job.scaling_job(ert, arguments)

    for index, node in enumerate(obs_vector):
        assert node.getStdScaling(index) == 1.0


@pytest.mark.equinor_test
@pytest.mark.usefixtures("setup_tmpdir")
def test_main_entry_point_block_data_calc():
    arguments = {"CALCULATE_KEYS": {"keys": [{"key": "RFT3"}]}}

    test_data_dir = os.path.join(_TEST_DATA_DIR, "Equinor", "config", "with_RFT")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("config")
    ert = EnKFMain(res_config)
    obs = ert.getObservations()

    obs_vector = obs["RFT3"]

    for index, node in enumerate(obs_vector):
        assert node.getStdScaling(index) == 1.0

    scaling_job.scaling_job(ert, arguments)

    for index, node in enumerate(obs_vector):
        assert node.getStdScaling(index) == 2.0


@pytest.mark.equinor_test
@pytest.mark.usefixtures("setup_tmpdir")
def test_main_entry_point_block_and_summary_data_calc():
    arguments = {"CALCULATE_KEYS": {"keys": [{"key": "FOPT"}, {"key": "RFT3"}]}}

    test_data_dir = os.path.join(_TEST_DATA_DIR, "Equinor", "config", "with_RFT")

    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("config")
    ert = EnKFMain(res_config)
    obs = ert.getObservations()

    obs_vector = obs["RFT3"]

    for index, node in enumerate(obs_vector):
        assert node.getStdScaling(index) == 1.0

    scaling_job.scaling_job(ert, arguments)

    for index, node in enumerate(obs_vector):
        assert node.getStdScaling(index) == np.sqrt(65)


@pytest.mark.usefixtures("setup_tmpdir")
def test_validate_failed_realizations():
    """
    Config has several failed realisations
    """
    test_data_dir = os.path.join(_TEST_DATA_DIR, "local", "custom_kw")
    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("mini_fail_config")
    ert = EnKFMain(res_config)
    observations = ert.getObservations()

    result = scaling_job.has_data(
        observations,
        ["GEN_PERLIN_1"],
        ert.getEnsembleSize(),
        ert.getEnkfFsManager().getCurrentFileSystem(),
    )
    assert result == []


@pytest.mark.usefixtures("setup_tmpdir")
def test_validate_no_realizations():
    """
    Ensamble has not run
    """
    test_data_dir = os.path.join(_TEST_DATA_DIR, "local", "poly_normal")
    shutil.copytree(test_data_dir, "test_data")
    os.chdir(os.path.join("test_data"))

    res_config = ResConfig("poly.ert")
    ert = EnKFMain(res_config)
    observations = ert.getObservations()

    result = scaling_job.has_data(
        observations,
        ["POLY_OBS"],
        ert.getEnsembleSize(),
        ert.getEnkfFsManager().getCurrentFileSystem(),
    )
    assert result == ["Key: POLY_OBS has no data"]
