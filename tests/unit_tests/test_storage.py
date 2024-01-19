import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List
from uuid import UUID

import hypothesis.strategies as st
import numpy as np
from hypothesis.extra.numpy import arrays
from hypothesis.stateful import Bundle, RuleBasedStateMachine, precondition, rule

from ert.config import EnkfObs, GenKwConfig, ParameterConfig, SurfaceConfig
from ert.config.enkf_observation_implementation_type import (
    EnkfObservationImplementationType,
)
from ert.config.general_observation import GenObservation
from ert.config.observation_vector import ObsVector
from ert.storage import open_storage

_CONFIG_DIR = tempfile.mkdtemp()
(Path(_CONFIG_DIR) / "template_file").write_text("")

parameter_configs = st.lists(
    st.one_of(
        st.builds(
            GenKwConfig,
            template_file=st.just(f"/{_CONFIG_DIR}/template_file"),
            transfer_function_definitions=st.just([]),
        ),
        st.builds(SurfaceConfig),
    )
)

words = st.text(
    min_size=1, max_size=8, alphabet=st.characters(min_codepoint=65, max_codepoint=90)
)


gen_observations = st.integers(min_value=1, max_value=10).flatmap(
    lambda size: st.builds(
        GenObservation,
        values=arrays(np.double, shape=size),
        stds=arrays(
            np.double,
            elements=st.floats(min_value=0.1, max_value=1.0),
            shape=size,
        ),
        indices=arrays(
            np.int64,
            elements=st.integers(min_value=0, max_value=100),
            shape=size,
        ),
        std_scaling=arrays(np.double, shape=size),
    )
)


@st.composite
def observation_dicts(draw):
    return {draw(st.integers(min_value=0, max_value=200)): draw(gen_observations)}


observations = st.builds(
    EnkfObs,
    obs_vectors=st.dictionaries(
        words,
        st.builds(
            ObsVector,
            observation_type=st.just(EnkfObservationImplementationType.GEN_OBS),
            observation_key=words,
            data_key=words,
            observations=observation_dicts(),
        ),
    ),
)

ensemble_sizes = st.integers(min_value=1, max_value=1000)


@dataclass(frozen=True)
class Experiment:
    uuid: UUID
    mount_point: Path


@dataclass(frozen=True)
class Ensemble:
    uuid: UUID
    mount_point: Path


class StatefulTest(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.ensemble_model = {}
        self.observations_model = {}
        self.storage = None
        self.tmpdir = None

    experiments = Bundle("experiments")
    ensembles = Bundle("ensembles")

    @precondition(lambda self: self.storage is None)
    @rule()
    def new(self):
        self.tmpdir = tempfile.mkdtemp()
        self.storage = open_storage(self.tmpdir, "w")
        self.ensemble_model = defaultdict(list)
        self.observations_model = defaultdict(dict)

    @precondition(lambda self: self.storage is not None)
    @rule(target=experiments, parameters=parameter_configs)
    def create_experiment(self, parameters: List[ParameterConfig]):
        assert self.storage is not None
        return self.storage.create_experiment(parameters).id

    @precondition(lambda self: self.storage is not None)
    @rule(target=experiments, parameters=parameter_configs, obs=observations)
    def create_experiment_with_observations(
        self, parameters: List[ParameterConfig], obs
    ):
        assert self.storage is not None
        experiment = self.storage.create_experiment(
            parameters, observations=obs.datasets
        ).id
        self.observations_model[experiment] = obs.datasets
        return experiment

    @precondition(lambda self: self.storage is not None)
    @rule(target=ensembles, experiment=experiments, ensemble_size=ensemble_sizes)
    def create_ensemble(self, experiment: UUID, ensemble_size: int):
        assert self.storage is not None
        ensemble = self.storage.create_ensemble(experiment, ensemble_size=ensemble_size)
        self.ensemble_model[experiment].append(ensemble.id)
        return ensemble.id

    @precondition(lambda self: self.storage is not None)
    @rule(id=experiments)
    def get_experiment(self, id: UUID):
        assert self.storage is not None
        experiment = self.storage.get_experiment(id)
        assert experiment.id == id
        assert self.ensemble_model[id] == [e.id for e in experiment.ensembles]
        assert sorted(self.observations_model[id].keys()) == sorted(
            experiment.observations.keys()
        )

    @precondition(lambda self: self.storage is not None)
    @rule(id=ensembles)
    def get_ensemble(self, id: UUID):
        assert self.storage is not None
        ensemble = self.storage.get_ensemble(id)
        assert ensemble.id == id

    def teardown(self):
        if self.storage is not None:
            self.storage.close()
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir)


TestStorage = StatefulTest.TestCase
