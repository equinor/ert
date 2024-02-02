import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch
from uuid import UUID

import hypothesis.strategies as st
import numpy as np
import pytest
import xarray as xr
from hypothesis.extra.numpy import arrays
from hypothesis.stateful import Bundle, RuleBasedStateMachine, initialize, rule

from ert.config import (
    EnkfObs,
    Field,
    GenDataConfig,
    GenKwConfig,
    ParameterConfig,
    ResponseConfig,
    SummaryConfig,
    SurfaceConfig,
)
from ert.config.enkf_observation_implementation_type import (
    EnkfObservationImplementationType,
)
from ert.config.general_observation import GenObservation
from ert.config.observation_vector import ObsVector
from ert.storage import StorageReader, open_storage
from ert.storage import local_storage as local
from ert.storage.realization_storage_state import RealizationStorageState
from tests.unit_tests.config.egrid_generator import egrids
from tests.unit_tests.config.summary_generator import summary_keys


def _cases(storage):
    return sorted(x.name for x in storage.ensembles)


def test_open_empty_reader(tmp_path):
    with open_storage(tmp_path / "empty", mode="r") as storage:
        assert _cases(storage) == []

    # StorageReader doesn't create an empty directory
    assert not (tmp_path / "empty").is_dir()


def test_open_empty_accessor(tmp_path):
    with open_storage(tmp_path / "empty", mode="w") as storage:
        assert _cases(storage) == []

    # StorageAccessor creates the directory
    assert (tmp_path / "empty").is_dir()


def test_refresh(tmp_path):
    with open_storage(tmp_path, mode="w") as accessor:
        experiment_id = accessor.create_experiment()
        with open_storage(tmp_path, mode="r") as reader:
            assert _cases(accessor) == _cases(reader)

            accessor.create_ensemble(experiment_id, name="foo", ensemble_size=42)
            # Reader does not know about the newly created ensemble
            assert _cases(accessor) != _cases(reader)

            reader.refresh()
            # Reader knows about it after the refresh
            assert _cases(accessor) == _cases(reader)


def test_runtime_types(tmp_path):
    with open_storage(tmp_path) as storage:
        assert isinstance(storage, local.LocalStorageReader)
        assert not isinstance(storage, local.LocalStorageAccessor)

    with open_storage(tmp_path, mode="r") as storage:
        assert isinstance(storage, local.LocalStorageReader)
        assert not isinstance(storage, local.LocalStorageAccessor)

    with open_storage(tmp_path, mode="w") as storage:
        assert isinstance(storage, local.LocalStorageReader)
        assert isinstance(storage, local.LocalStorageAccessor)


def test_to_accessor(tmp_path):
    """
    Type-correct casting from StorageReader to StorageAccessor in cases where a
    function accepts StorageReader, but has additional functionality if it's a
    StorageAccessor. Eg, in the ERT GUI, we may pass StorageReader to the
    CaseList widget, which lists which ensembles are available, but if
    .to_accessor() doesn't throw then CaseList can also create new ensembles.
    """

    with open_storage(tmp_path) as storage_reader, pytest.raises(TypeError):
        storage_reader.to_accessor()

    with open_storage(tmp_path, mode="w") as storage_accessor:
        storage_reader: StorageReader = storage_accessor
        storage_reader.to_accessor()


parameter_configs = st.lists(
    st.one_of(
        st.builds(
            GenKwConfig,
            template_file=st.just(None),
            transfer_function_definitions=st.just([]),
        ),
        st.builds(SurfaceConfig),
    ),
    unique_by=lambda x: x.name,
)

response_configs = st.lists(
    st.one_of(
        st.builds(
            GenDataConfig,
        ),
        st.builds(
            SummaryConfig,
            name=st.text(),
            input_file=st.text(
                alphabet=st.characters(min_codepoint=65, max_codepoint=90)
            ),
            keys=st.lists(summary_keys, max_size=3),
            refcase=st.just(None),
        ),
    ),
    unique_by=lambda x: x.name,
)

ensemble_sizes = st.integers(min_value=1, max_value=1000)
coordinates = st.integers(min_value=1, max_value=100)


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


@dataclass
class Experiment:
    ensembles: Dict[UUID, Dict[str, Any]] = field(default_factory=dict)
    parameters: List[ParameterConfig] = field(default_factory=list)
    responses: List[ResponseConfig] = field(default_factory=list)
    observations: Dict[str, xr.Dataset] = field(default_factory=dict)


@st.composite
def fields(draw, egrid) -> List[Field]:
    grid_file, grid = egrid
    nx, ny, nz = grid.shape
    return [
        draw(
            st.builds(
                Field,
                name=st.just(f"Field{i}"),
                file_format=st.just("roff_binary"),
                grid_file=st.just(grid_file),
                nx=st.just(nx),
                ny=st.just(ny),
                nz=st.just(nz),
                output_file=st.just(Path(f"field{i}.roff")),
            )
        )
        for i in range(10)
    ]


class StatefulTest(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.tmpdir = tempfile.mkdtemp()
        self.storage = open_storage(self.tmpdir + "/storage/", "w")
        self.experiments = defaultdict(Experiment)
        self.failure_messages = {}
        assert list(self.storage.ensembles) == []

    experiment_ids = Bundle("experiments")
    ensemble_ids = Bundle("ensembles")
    failures = Bundle("failures")
    field_list = Bundle("field_list")
    grid = Bundle("grid")

    @initialize(target=grid, egrid=egrids)
    def create_grid(self, egrid):
        grid_file = self.tmpdir + "/grid.egrid"
        egrid.to_file(grid_file)
        return (grid_file, egrid)

    @initialize(
        target=field_list,
        fields=grid.flatmap(fields),
    )
    def create_field_list(self, fields):
        return fields

    @rule()
    def double_open_timeout(self):
        # Opening with write access will timeout when opening lock
        with patch(
            "ert.storage.local_storage.LocalStorageAccessor.LOCK_TIMEOUT", 0.0
        ), pytest.raises(TimeoutError):
            open_storage(self.tmpdir + "/storage/", mode="w")

    @rule()
    def reopen(self):
        cases = sorted(e.id for e in self.storage.ensembles)
        self.storage.close()
        self.storage = open_storage(self.tmpdir + "/storage/", mode="w")
        assert cases == sorted(e.id for e in self.storage.ensembles)

    @rule(
        target=experiment_ids,
        parameters=st.one_of(parameter_configs, field_list),
        responses=response_configs,
        obs=observations,
    )
    def create_experiment(
        self,
        parameters: List[ParameterConfig],
        responses: List[ResponseConfig],
        obs: EnkfObs,
    ):
        experiment_id = self.storage.create_experiment(
            parameters=parameters, responses=responses, observations=obs.datasets
        ).id
        self.experiments[experiment_id].parameters = parameters
        self.experiments[experiment_id].responses = responses
        self.experiments[experiment_id].observations = obs.datasets

        # Ensure that there is at least one ensemble in the experiment
        # to avoid https://github.com/equinor/ert/issues/7040
        ensemble = self.storage.create_ensemble(experiment_id, ensemble_size=1)
        self.experiments[experiment_id].ensembles[ensemble.id] = {}

        return experiment_id

    @rule(
        ensemble_id=ensemble_ids,
        field_data=grid.flatmap(lambda g: arrays(np.float32, shape=g[1].shape)),
    )
    def save_field(self, ensemble_id: UUID, field_data):
        ensemble = self.storage.get_ensemble(ensemble_id)
        experiment_id = ensemble.experiment_id
        parameters = self.experiments[experiment_id].parameters
        fields = [p for p in parameters if isinstance(p, Field)]
        for f in fields:
            self.experiments[experiment_id].ensembles[ensemble_id][f.name] = field_data
            ensemble.save_parameters(
                f.name,
                1,
                xr.DataArray(
                    field_data,
                    name="values",
                    dims=["x", "y", "z"],  # type: ignore
                ).to_dataset(),
            )

    @rule(
        ensemble_id=ensemble_ids,
    )
    def get_field(self, ensemble_id: UUID):
        ensemble = self.storage.get_ensemble(ensemble_id)
        experiment_id = ensemble.experiment_id
        field_names = self.experiments[experiment_id].ensembles[ensemble_id].keys()
        for f in field_names:
            field_data = ensemble.load_parameters(f, 1)
            np.testing.assert_array_equal(
                self.experiments[experiment_id].ensembles[ensemble_id][f],
                field_data["values"],
            )

    @rule(
        target=ensemble_ids,
        experiment=experiment_ids,
        ensemble_size=ensemble_sizes,
    )
    def create_ensemble(self, experiment: UUID, ensemble_size: int):
        ensemble = self.storage.create_ensemble(experiment, ensemble_size=ensemble_size)
        assert ensemble in self.storage.ensembles
        self.experiments[experiment].ensembles[ensemble.id] = {}

        # https://github.com/equinor/ert/issues/7046
        # assert (
        #    ensemble.get_ensemble_state()
        #    == [RealizationStorageState.UNDEFINED] * ensemble_size
        # )

        return ensemble.id

    @rule(
        target=ensemble_ids,
        prior=ensemble_ids,
    )
    def create_ensemble_from_prior(self, prior: UUID):
        prior_ensemble = self.storage.get_ensemble(prior)
        experiment = prior_ensemble.experiment_id
        size = prior_ensemble.ensemble_size
        ensemble = self.storage.create_ensemble(
            experiment, ensemble_size=size, prior_ensemble=prior
        )
        assert ensemble in self.storage.ensembles
        self.experiments[experiment].ensembles[ensemble.id] = {}
        # https://github.com/equinor/ert/issues/7046
        # assert (
        #    ensemble.get_ensemble_state()
        #    == [RealizationStorageState.PARENT_FAILURE] * size
        # )

        return ensemble.id

    @rule(id=experiment_ids)
    def get_experiment(self, id: UUID):
        experiment = self.storage.get_experiment(id)
        assert experiment.id == id
        assert sorted(self.experiments[id].ensembles) == sorted(
            e.id for e in experiment.ensembles
        )
        assert (
            list(experiment.response_configuration.values())
            == self.experiments[id].responses
        )
        assert self.experiments[id].observations == pytest.approx(
            experiment.observations
        )

    @rule(id=ensemble_ids)
    def get_ensemble(self, id: UUID):
        ensemble = self.storage.get_ensemble(id)
        assert ensemble.id == id

    @rule(target=failures, id=ensemble_ids, data=st.data(), message=st.text())
    def set_failure(self, id: UUID, data: st.DataObject, message: str):
        ensemble = self.storage.get_ensemble(id)
        assert ensemble.id == id

        realization = data.draw(
            st.integers(min_value=0, max_value=ensemble.ensemble_size - 1)
        )

        ensemble.set_failure(
            realization, RealizationStorageState.PARENT_FAILURE, message
        )
        self.failure_messages[ensemble.id, realization] = message

        return (ensemble.id, realization)

    @rule(failure=failures)
    def get_failure(self, failure):
        (ensemble, realization) = failure
        fail = self.storage.get_ensemble(ensemble).get_failure(realization)
        assert fail is not None
        assert fail.message == self.failure_messages[ensemble, realization]

    def teardown(self):
        if self.storage is not None:
            self.storage.close()
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir)


TestStorage = StatefulTest.TestCase
