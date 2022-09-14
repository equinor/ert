#  Copyright (C) 2012  Equinor ASA, Norway.
#
#  The file 'test_enkf.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.
from textwrap import dedent

import pytest

from ert._c_wrappers.enkf import (
    AnalysisConfig,
    EclConfig,
    EnkfFs,
    EnkfObs,
    EnsembleConfig,
    ModelConfig,
    ObsVector,
    ResConfig,
    RunArg,
    SiteConfig,
)
from ert._c_wrappers.enkf.config import EnkfConfigNode
from ert._c_wrappers.enkf.enkf_main import EnKFMain
from ert._c_wrappers.enkf.enums import (
    EnkfObservationImplementationType,
    LoadFailTypeEnum,
)
from ert._c_wrappers.enkf.observations.summary_observation import SummaryObservation


def test_repr(minimum_case):
    assert repr(minimum_case).startswith("EnKFMain(ensemble_size")


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
        assert res_config.rng_config.random_seed == str(random_seed)
