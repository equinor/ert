import logging
import os
from pathlib import Path
from textwrap import dedent
from unittest.mock import MagicMock

import pytest
from ecl.summary import EclSum

from ert._c_wrappers.enkf import (
    AnalysisConfig,
    EclConfig,
    EnkfFs,
    EnKFMain,
    EnkfObs,
    EnsembleConfig,
    ModelConfig,
    ObsVector,
    ResConfig,
    RunArg,
    SiteConfig,
    TimeMap,
)
from ert._c_wrappers.enkf.config import EnkfConfigNode
from ert._c_wrappers.enkf.enums import (
    EnkfObservationImplementationType,
    LoadFailTypeEnum,
)
from ert._c_wrappers.enkf.enums.realization_state_enum import RealizationStateEnum
from ert._c_wrappers.enkf.observations.summary_observation import SummaryObservation


@pytest.mark.unstable
def test_ecl_config_creation(minimum_case):
    assert isinstance(minimum_case.analysisConfig(), AnalysisConfig)
    assert isinstance(minimum_case.eclConfig(), EclConfig)

    with pytest.raises(AssertionError):  # Null pointer!
        assert isinstance(minimum_case.eclConfig().getRefcase(), EclSum)

    file_system = minimum_case.getEnkfFsManager().getCurrentFileSystem()
    assert file_system.getCaseName() == "default"
    time_map = file_system.getTimeMap()
    assert isinstance(time_map, TimeMap)


@pytest.fixture
def enkf_main(tmp_path):
    (tmp_path / "test.ert").write_text("NUM_REALIZATIONS 1\nJOBNAME name%d")
    os.chdir(tmp_path)
    yield EnKFMain(ResConfig("test.ert"))


def test_create_ensemble_experiment_run_context(enkf_main):
    fs = MagicMock()

    enkf_main._create_run_context = MagicMock()

    realizations = [True] * 10
    iteration = 0

    enkf_main.create_ensemble_experiment_run_context(
        active_mask=realizations, source_filesystem=fs, iteration=iteration
    )

    enkf_main._create_run_context.assert_called_with(
        iteration=iteration,
        active_mask=realizations,
        source_filesystem=fs,
        target_fs=None,
    )


def test_create_ensemble_smoother_run_context(enkf_main):
    fs = MagicMock()
    fs2 = MagicMock()

    enkf_main._create_run_context = MagicMock()

    realizations = [True] * 10
    iteration = 0

    enkf_main.create_ensemble_smoother_run_context(
        active_mask=realizations,
        source_filesystem=fs,
        target_filesystem=fs2,
        iteration=iteration,
    )

    enkf_main._create_run_context.assert_called_with(
        iteration=iteration,
        active_mask=realizations,
        source_filesystem=fs,
        target_fs=fs2,
    )


def test_create_run_context(monkeypatch, enkf_main):

    iteration = 0
    ensemble_size = 10

    run_context = enkf_main._create_run_context(
        iteration=iteration, active_mask=[True] * ensemble_size
    )
    assert run_context.sim_fs == enkf_main.getCurrentFileSystem()
    assert run_context.target_fs == enkf_main.getCurrentFileSystem()
    assert run_context.mask == [True] * ensemble_size
    assert run_context.paths == [
        f"{Path().absolute()}/simulations/realization{i}" for i in range(ensemble_size)
    ]
    assert run_context.jobnames == [f"name{i}" for i in range(ensemble_size)]

    substitutions = enkf_main.substituter.get_substitutions(1, iteration)
    assert "<RUNPATH>" in substitutions
    assert substitutions["<ECL_BASE>"] == "name1"
    assert substitutions["<ECLBASE>"] == "name1"
    assert substitutions["<ITER>"] == str(iteration)
    assert substitutions["<IENS>"] == "1"


def test_create_set_geo_id(enkf_main):

    iteration = 1
    realization = 2
    geo_id = "geo_id"

    enkf_main.set_geo_id("geo_id", realization, iteration)

    assert (
        enkf_main.substituter.get_substitutions(realization, iteration)["<GEO_ID>"]
        == geo_id
    )


@pytest.mark.usefixtures("use_tmpdir")
def test_that_current_case_file_is_written():
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        JOBNAME my_case%d
        """
    )
    Path("config.ert").write_text(config_text)
    res_config = ResConfig("config.ert")
    ert = EnKFMain(res_config)
    new_fs = EnkfFs.createFileSystem("new_fs", True, ert._ensemble_size)
    ert.switchFileSystem(new_fs)
    assert (Path("storage") / "current_case").read_text() == "new_fs"


@pytest.mark.usefixtures("use_tmpdir")
def test_that_current_case_file_can_have_newline():
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        JOBNAME my_case%d
        """
    )
    Path("config.ert").write_text(config_text)
    res_config = ResConfig("config.ert")
    EnKFMain(res_config)
    assert (Path("storage") / "current_case").read_text() == "default"
    del res_config
    (Path("storage") / "current_case").write_text("default\n")
    res_config = ResConfig("config.ert")
    EnKFMain(res_config)


def test_assert_symlink_deleted(snake_oil_field_example):
    ert = snake_oil_field_example

    # create directory structure
    run_context = ert.create_ensemble_experiment_run_context(iteration=0)
    ert.createRunPath(run_context)

    # replace field file with symlink
    linkpath = f"{run_context[0].runpath}/permx.grdcel"
    targetpath = f"{run_context[0].runpath}/permx.grdcel.target"
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


def test_site_bootstrap():
    with pytest.raises(TypeError):
        EnKFMain(None)


def test_invalid_res_config():
    with pytest.raises(TypeError):
        EnKFMain(res_config="This is not a ResConfig instance")


def test_invalid_parameter_count_2_res_config():
    with pytest.raises(ValueError):
        ResConfig(user_config_file="a", config="b")


def test_invalid_parameter_count_3_res_config():
    with pytest.raises(ValueError):
        ResConfig(user_config_file="a", config="b", config_dict="c")


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


def test_config(minimum_case):

    assert isinstance(minimum_case.ensembleConfig(), EnsembleConfig)
    assert isinstance(minimum_case.analysisConfig(), AnalysisConfig)
    assert isinstance(minimum_case.getModelConfig(), ModelConfig)
    assert isinstance(minimum_case.siteConfig(), SiteConfig)
    assert isinstance(minimum_case.eclConfig(), EclConfig)

    assert isinstance(minimum_case.getObservations(), EnkfObs)
    assert isinstance(minimum_case.getEnkfFsManager().getCurrentFileSystem(), EnkfFs)

    assert minimum_case.getMountPoint().endswith("/Ensemble")


def test_run_context(minimum_case):
    fs_manager = minimum_case.getEnkfFsManager()
    fs = fs_manager.getCurrentFileSystem()
    iactive = [True] * 10
    iactive[0] = False
    iactive[1] = False
    run_context = minimum_case.create_ensemble_experiment_run_context(
        source_filesystem=fs, active_mask=iactive, iteration=0
    )

    assert len(run_context) == 10

    with pytest.raises(IndexError):
        _ = run_context[10]

    with pytest.raises(TypeError):
        _ = run_context["String"]

    assert not run_context.is_active(0)
    run_arg = run_context[2]
    assert isinstance(run_arg, RunArg)

    rng1 = minimum_case.rng()
    rng1.setState("ABCDEFGHIJ012345")
    d1 = rng1.getDouble()
    rng1.setState("ABCDEFGHIJ012345")
    rng2 = minimum_case.rng()
    d2 = rng2.getDouble()
    assert d1 == d2


@pytest.mark.parametrize(
    "random_seed", ["0", "1234", "123ABC", "123456789ABCDEFGHIJKLMNOPGRST", "123456"]
)
def test_random_seed_initialization_of_rngs(random_seed, tmpdir):
    """
    This is a regression test to make sure the seed can be sampled correctly,
    and that it wraps on int32 overflow.
    """
    with tmpdir.as_cwd():
        config = dedent(
            f"""
        JOBNAME my_name%d
        NUM_REALIZATIONS 10
        RANDOM_SEED {random_seed}
        """
        )
        with open("config.ert", "w") as fh:
            fh.writelines(config)

        res_config = ResConfig("config.ert")
        EnKFMain(res_config)
        assert res_config.random_seed == str(random_seed)


def test_failed_realizations(setup_case):
    """mini_fail_config has the following realization success/failures:

    0 OK
    1 GenData report step 1 missing
    2 GenData report step 2 missing, Forward Model Component Target File not found
    3 GenData report step 3 missing, Forward Model Component Target File not found
    4 GenData report step 1 missing
    5 GenData report step 2 missing, Forward Model Component Target File not found
    6 GenData report step 3 missing
    7 Forward Model Target File not found.
    8 OK
    9 OK
    """
    ert = EnKFMain(setup_case("mini_ert", "mini_fail_config"))
    fs = ert.getEnkfFsManager().getCurrentFileSystem()

    realizations_list = fs.realizationList(RealizationStateEnum.STATE_HAS_DATA)
    assert 0 in realizations_list
    assert 8 in realizations_list
    assert 9 in realizations_list

    realizations_list = fs.realizationList(RealizationStateEnum.STATE_LOAD_FAILURE)
    assert 1 in realizations_list
    assert 2 in realizations_list
    assert 3 in realizations_list
    assert 4 in realizations_list
    assert 5 in realizations_list
    assert 6 in realizations_list
    assert 7 in realizations_list


@pytest.mark.usefixtures("use_tmpdir")
def test_data_kw():
    # Write a minimal config file with DEFINE
    with open("config_file.ert", "w") as fout:
        fout.write("NUM_REALIZATIONS 1\nDEFINE MY_PATH <CONFIG_PATH>")
    res_config = ResConfig("config_file.ert")
    ert = EnKFMain(res_config)
    data_kw = ert.getDataKW()
    my_path = data_kw["MY_PATH"]
    assert my_path == os.getcwd()


def test_load_results_manually(setup_case):
    res_config = setup_case("mini_ert", "mini_fail_config")
    ert = EnKFMain(res_config)
    load_into_case = "A1"
    load_from_case = "default_1"

    load_into = ert.getEnkfFsManager().getFileSystem(load_into_case)
    load_from = ert.getEnkfFsManager().getFileSystem(load_from_case)

    ert.getEnkfFsManager().switchFileSystem(load_from)
    realisations = [True] * 10
    realisations[7] = False
    iteration = 0

    loaded = ert.loadFromForwardModel(realisations, iteration, load_into)

    load_into_case_state_map = load_into.getStateMap()

    load_into_states = list(load_into_case_state_map)

    expected = [
        RealizationStateEnum.STATE_HAS_DATA,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_LOAD_FAILURE,
        RealizationStateEnum.STATE_UNDEFINED,
        RealizationStateEnum.STATE_HAS_DATA,
        RealizationStateEnum.STATE_HAS_DATA,
    ]

    assert load_into_states == expected
    assert loaded == 3


@pytest.mark.skip
@pytest.mark.parametrize("lazy_load", [True, False])
def test_load_results_manually2(setup_case, caplog, monkeypatch, lazy_load):
    """
    This little test does not depend on Equinor-data and only verifies
    the lazy_load flag in forward_load_context plus memory-logging
    """
    if lazy_load:
        monkeypatch.setenv("ERT_LAZY_LOAD_SUMMARYDATA", str(lazy_load))
    res_config = setup_case("snake_oil", "snake_oil.ert")
    ert = EnKFMain(res_config)
    load_from = ert.getEnkfFsManager().getFileSystem("default_0")
    ert.getEnkfFsManager().switchFileSystem(load_from)
    realisations = [False] * 25
    realisations[0] = True  # only need one to test what we want
    with caplog.at_level(logging.INFO):
        loaded = ert.loadFromForwardModel(realisations, 0, load_from)
        assert loaded == 0  # they will in fact all fail, but that's ok
        assert f"lazy={lazy_load}".lower() in caplog.text
