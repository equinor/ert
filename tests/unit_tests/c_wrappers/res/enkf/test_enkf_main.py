import os
from pathlib import Path
from textwrap import dedent

import pytest
from ecl.summary import EclSum

from ert._c_wrappers.enkf import (
    AnalysisConfig,
    EnKFMain,
    EnkfObs,
    EnsembleConfig,
    ModelConfig,
    ObservationConfigError,
    ObsVector,
    ResConfig,
)
from ert._c_wrappers.enkf.config import EnkfConfigNode
from ert._c_wrappers.enkf.enums import (
    EnkfObservationImplementationType,
    LoadFailTypeEnum,
)
from ert._c_wrappers.enkf.observations.summary_observation import SummaryObservation


@pytest.mark.unstable
def test_ecl_config_creation(minimum_case):
    assert isinstance(minimum_case.analysisConfig(), AnalysisConfig)
    assert isinstance(minimum_case.ensembleConfig(), EnsembleConfig)

    with pytest.raises(AssertionError):  # Null pointer!
        assert isinstance(minimum_case.ensembleConfig().refcase, EclSum)


@pytest.fixture(name="enkf_main")
def enkf_main_fixture(tmp_path, monkeypatch):
    (tmp_path / "test.ert").write_text("NUM_REALIZATIONS 1\nJOBNAME name%d")
    monkeypatch.chdir(tmp_path)
    yield EnKFMain(ResConfig("test.ert"))


def test_create_run_context(monkeypatch, enkf_main, prior_ensemble):
    iteration = 0
    ensemble_size = 10

    run_context = enkf_main.ensemble_context(
        prior_ensemble, [True] * ensemble_size, iteration=iteration
    )
    assert run_context.sim_fs.name == "prior"
    assert run_context.mask == [True] * ensemble_size
    assert [real.runpath for real in run_context] == [
        f"{Path().absolute()}/simulations/realization-{i}/iter-0"
        for i in range(ensemble_size)
    ]
    assert [real.job_name for real in run_context] == [
        f"name{i}" for i in range(ensemble_size)
    ]

    substitutions = run_context.substituter
    assert "<RUNPATH>" in substitutions
    assert substitutions.get("<ECL_BASE>") == "name<IENS>"
    assert substitutions.get("<ECLBASE>") == "name<IENS>"


def test_assert_symlink_deleted(snake_oil_field_example, prior_ensemble):
    ert = snake_oil_field_example

    # create directory structure
    run_context = ert.ensemble_context(
        prior_ensemble, [True] * (ert.getEnsembleSize()), iteration=0
    )
    ert.sample_prior(run_context.sim_fs, run_context.active_realizations)
    ert.createRunPath(run_context)

    # replace field file with symlink
    linkpath = f"{run_context[0].runpath}/permx.grdecl"
    targetpath = f"{run_context[0].runpath}/permx.grdecl.target"
    with open(targetpath, "a", encoding="utf-8"):
        pass
    os.remove(linkpath)
    os.symlink(targetpath, linkpath)

    # recreate directory structure
    ert.createRunPath(run_context)

    # ensure field symlink is replaced by file
    assert not os.path.islink(linkpath)


def test_repr(minimum_case):
    assert repr(minimum_case).startswith("EnKFMain(size: 10, config")


def test_bootstrap(minimum_case):
    assert bool(minimum_case)


def test_invalid_res_config():
    with pytest.raises(TypeError):
        # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
        EnKFMain(res_config="This is not a ResConfig instance")


def test_invalid_parameter_count_2_res_config():
    with pytest.raises(ValueError):
        ResConfig(user_config_file="a", config_dict="b")


def test_observations(minimum_case):
    count = 10
    summary_key = "test_key"
    observation_key = "test_obs_key"
    summary_observation_node = EnkfConfigNode.createSummaryConfigNode(
        summary_key, LoadFailTypeEnum.LOAD_FAIL_EXIT
    )
    observation_vector = ObsVector(
        EnkfObservationImplementationType.SUMMARY_OBS,
        observation_key,
        summary_observation_node,
        count,
    )

    minimum_case.getObservations().addObservationVector(observation_vector)

    values = []
    for index in range(0, count):
        value = index * 10.5
        std = index / 10.0
        summary_observation_node = SummaryObservation(
            summary_key, observation_key, value, std
        )
        observation_vector.installNode(index, summary_observation_node)
        assert observation_vector.getNode(index) == summary_observation_node
        assert summary_observation_node.getValue() == value
        values.append((index, value, std))

    observations = minimum_case.getObservations()
    test_vector = observations[observation_key]
    index = 0
    for node in test_vector:
        assert isinstance(node, SummaryObservation)
        assert node.getValue() == index * 10.5
        index += 1

    assert observation_vector == test_vector
    for index, value, std in values:
        assert test_vector.isActive(index)

        summary_observation_node = test_vector.getNode(index)

        assert summary_observation_node.getValue() == value
        assert summary_observation_node.getStandardDeviation() == std
        assert summary_observation_node.getSummaryKey() == summary_key


def test_that_empty_observations_file_causes_exception(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 10
        OBS_CONFIG observations
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fh:
            fh.writelines("")

        res_config = ResConfig("config.ert")

        with pytest.raises(
            expected_exception=ObservationConfigError,
            match="Empty observations file",
        ):
            EnKFMain(res_config)


def test_that_having_no_refcase_but_history_observations_causes_exception(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 10
        OBS_CONFIG observations
        TIME_MAP time_map.txt
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines("HISTORY_OBSERVATION FOPR;")
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")

        res_config = ResConfig("config.ert")

        with pytest.raises(
            expected_exception=ObservationConfigError,
            match="REFCASE is required for HISTORY_OBSERVATION",
        ):
            EnKFMain(res_config)


def test_that_missing_obs_file_raises_exception(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 10
        OBS_CONFIG observations
        TIME_MAP time_map.txt
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                [
                    "GENERAL_OBSERVATION OBS {",
                    "   DATA       = RES;",
                    "   INDEX_LIST = 0,2,4,6,8;",
                    "   RESTART    = 0;",
                    "   OBS_FILE   = obs_data.txt;",
                    "};",
                ]
            )
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")

        res_config = ResConfig("config.ert")

        with pytest.raises(
            expected_exception=ObservationConfigError,
            match="did not resolve to a valid path:\n OBS_FILE",
        ):
            EnKFMain(res_config)


def test_that_missing_time_map_raises_exception(tmpdir):
    with tmpdir.as_cwd():
        config = dedent(
            """
        JOBNAME my_name%d
        NUM_REALIZATIONS 10
        OBS_CONFIG observations
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config)
        with open("observations", "w", encoding="utf-8") as fo:
            fo.writelines(
                [
                    "GENERAL_OBSERVATION OBS {",
                    "   DATA       = RES;",
                    "   INDEX_LIST = 0,2,4,6,8;",
                    "   RESTART    = 0;",
                    "   OBS_FILE   = obs_data.txt;",
                    "};",
                ]
            )
        with open("time_map.txt", "w", encoding="utf-8") as fo:
            fo.writelines("2023-02-01")

        res_config = ResConfig("config.ert")

        with pytest.raises(
            expected_exception=ObservationConfigError,
            match="Incorrect observations file",
        ):
            EnKFMain(res_config)


def test_config(minimum_case):
    assert isinstance(minimum_case.ensembleConfig(), EnsembleConfig)
    assert isinstance(minimum_case.analysisConfig(), AnalysisConfig)
    assert isinstance(minimum_case.getModelConfig(), ModelConfig)

    assert isinstance(minimum_case.getObservations(), EnkfObs)


@pytest.mark.parametrize(
    "random_seed", ["0", "1234", "123ABC", "123456789ABCDEFGHIJKLMNOPGRST", "123456"]
)
def test_random_seed_initialization_of_rngs(random_seed, tmpdir):
    """
    This is a regression test to make sure the seed can be sampled correctly,
    and that it wraps on int32 overflow.
    """
    with tmpdir.as_cwd():
        config_content = dedent(
            f"""
        JOBNAME my_name%d
        NUM_REALIZATIONS 10
        RANDOM_SEED {random_seed}
        """
        )
        with open("config.ert", "w", encoding="utf-8") as fh:
            fh.writelines(config_content)

        res_config = ResConfig("config.ert")
        EnKFMain(res_config)
        assert res_config.random_seed == str(random_seed)


@pytest.mark.usefixtures("use_tmpdir")
def test_ert_context():
    # Write a minimal config file with DEFINE
    with open("config_file.ert", "w", encoding="utf-8") as fout:
        fout.write("NUM_REALIZATIONS 1\nDEFINE MY_PATH <CONFIG_PATH>")
    res_config = ResConfig("config_file.ert")
    ert = EnKFMain(res_config)
    context = ert.get_context()
    my_path = context["MY_PATH"]
    assert my_path == os.getcwd()
