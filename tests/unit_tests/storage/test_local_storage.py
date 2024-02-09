import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch
from uuid import UUID

import hypothesis.strategies as st
import numpy as np
import pytest
import xarray as xr
from hypothesis import assume
from hypothesis.extra.numpy import arrays
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    consumes,
    initialize,
    rule,
)

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
from ert.storage import open_storage
from ert.storage.local_storage import (
    _LOCAL_STORAGE_VERSION,
    _is_block_storage,
    _storage_version,
)
from ert.storage.mode import ModeError
from ert.storage.realization_storage_state import RealizationStorageState
from tests.unit_tests.config.egrid_generator import egrids
from tests.unit_tests.config.summary_generator import summary_keys


def _cases(storage):
    return sorted(x.name for x in storage.ensembles)


def test_that_loading_parameter_via_response_api_fails(tmp_path):
    uniform_parameter = GenKwConfig(
        name="PARAMETER",
        forward_init=False,
        template_file="",
        transfer_function_definitions=[
            "KEY1 UNIFORM 0 1",
        ],
        output_file="kw.txt",
    )
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(
            parameters=[uniform_parameter],
        )
        prior = storage.create_ensemble(
            experiment,
            ensemble_size=1,
            iteration=0,
            name="prior",
        )

        prior.save_parameters(
            "PARAMETER",
            0,
            xr.Dataset(
                {
                    "values": ("names", [1.0]),
                    "transformed_values": ("names", [1.0]),
                    "names": ["KEY_1"],
                }
            ),
        )
        with pytest.raises(ValueError, match="PARAMETER is not a response"):
            prior.load_responses("PARAMETER", (0,))


def test_that_load_responses_throws_exception(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(experiment, name="foo", ensemble_size=1)

        with pytest.raises(
            expected_exception=ValueError, match="I_DONT_EXIST is not a response"
        ):
            ensemble.load_responses("I_DONT_EXIST", (1,))


def test_that_load_parameters_throws_exception(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(experiment, name="foo", ensemble_size=1)

        with pytest.raises(expected_exception=KeyError):
            ensemble.load_parameters("I_DONT_EXIST", 1)


def test_open_empty_read(tmp_path):
    with open_storage(tmp_path / "empty", mode="r") as storage:
        assert _cases(storage) == []

    # Storage doesn't create an empty directory
    assert not (tmp_path / "empty").is_dir()


def test_open_empty_write(tmp_path):
    with open_storage(tmp_path / "empty", mode="w") as storage:
        assert _cases(storage) == []

    # Storage creates the directory
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


def test_writing_to_read_only_storage_raises(tmp_path):
    with open_storage(tmp_path, mode="r") as storage, pytest.raises(ModeError):
        storage.create_experiment()


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
class Ensemble:
    uuid: UUID
    parameter_values: Dict[str, Any] = field(default_factory=dict)
    failure_messages: Dict[int, str] = field(default_factory=dict)


@dataclass
class Experiment:
    uuid: UUID
    ensembles: Dict[UUID, Ensemble] = field(default_factory=dict)
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
        assert _storage_version(Path(self.tmpdir + "/storage/")) is None
        self.storage = open_storage(self.tmpdir + "/storage/", "w")
        self.model = {}
        self.deleted_ensembles = {}
        assert list(self.storage.ensembles) == []

    experiments = Bundle("experiments")
    ensembles = Bundle("ensembles")
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
            "ert.storage.local_storage.LocalStorage.LOCK_TIMEOUT", 0.0
        ), pytest.raises(TimeoutError):
            open_storage(self.tmpdir + "/storage/", mode="w")

    @rule()
    def reopen(self):
        cases = sorted(e.id for e in self.storage.ensembles)
        self.storage.close()
        assert not _is_block_storage(Path(self.tmpdir + "/storage/"))
        assert (
            _storage_version(Path(self.tmpdir + "/storage/")) == _LOCAL_STORAGE_VERSION
        )
        self.storage = open_storage(self.tmpdir + "/storage/", mode="w")
        assert cases == sorted(e.id for e in self.storage.ensembles)

    @rule(
        target=experiments,
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
        model_experiment = Experiment(experiment_id)
        model_experiment.parameters = parameters
        model_experiment.responses = responses
        model_experiment.observations = obs.datasets

        # Ensure that there is at least one ensemble in the experiment
        # to avoid https://github.com/equinor/ert/issues/7040
        ensemble = self.storage.create_ensemble(experiment_id, ensemble_size=1)
        model_experiment.ensembles[ensemble.id] = Ensemble(ensemble.id)

        self.model[model_experiment.uuid] = model_experiment

        return model_experiment

    @rule(
        model_ensemble=ensembles,
        field_data=grid.flatmap(lambda g: arrays(np.float32, shape=g[1].shape)),
    )
    def save_field(self, model_ensemble: Ensemble, field_data):
        if model_ensemble.uuid not in self.deleted_ensembles:
            storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
            parameters = model_ensemble.parameter_values.values()
            fields = [p for p in parameters if isinstance(p, Field)]
            for f in fields:
                model_ensemble.parameter_values[f.name] = field_data
                storage_ensemble.save_parameters(
                    f.name,
                    1,
                    xr.DataArray(
                        field_data,
                        name="values",
                        dims=["x", "y", "z"],  # type: ignore
                    ).to_dataset(),
                )
        else:
            ensemble = self.deleted_ensembles[model_ensemble.uuid]
            parameters = model_ensemble.parameter_values.values()
            fields = [p for p in parameters if isinstance(p, Field)]
            for f in fields:
                model_ensemble.parameter_values[f.name] = field_data
                with pytest.raises(ModeError):
                    ensemble.save_parameters(
                        "param", 0, xr.Dataset({"values": [1, 2, 3]})
                    )

    @rule(
        model_ensemble=ensembles,
    )
    def get_field(self, model_ensemble: Ensemble):
        if model_ensemble.uuid not in self.deleted_ensembles:
            storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
            for f in model_ensemble.parameter_values:
                field_data = storage_ensemble.load_parameters(f, 1)
                np.testing.assert_array_equal(
                    model_ensemble.parameter_values[f],
                    field_data["values"],
                )
        else:
            ensemble = self.deleted_ensembles[model_ensemble.uuid]
            for f in model_ensemble.parameter_values:
                with pytest.raises(ModeError):
                    _ = ensemble.load_parameters(f, 1)

    @rule(model_ensemble=ensembles, parameter=words)
    def load_unknown_parameter(self, model_ensemble: Ensemble, parameter: str):
        assume(model_ensemble.uuid not in self.deleted_ensembles)
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        experiment_id = storage_ensemble.experiment_id
        parameter_names = [p.name for p in self.model[experiment_id].parameters]
        assume(parameter not in parameter_names)
        with pytest.raises(
            KeyError, match=f"No dataset '{parameter}' in storage for realization 0"
        ):
            _ = storage_ensemble.load_parameters(parameter, 0)

    @rule(
        target=ensembles,
        model_experiment=experiments,
        ensemble_size=ensemble_sizes,
    )
    def create_ensemble(self, model_experiment: Experiment, ensemble_size: int):
        ensemble = self.storage.create_ensemble(
            model_experiment.uuid, ensemble_size=ensemble_size
        )
        assert ensemble in self.storage.ensembles
        model_ensemble = Ensemble(ensemble.id)
        model_experiment.ensembles[ensemble.id] = model_ensemble

        # https://github.com/equinor/ert/issues/7046
        # assert (
        #    ensemble.get_ensemble_state()
        #    == [RealizationStorageState.UNDEFINED] * ensemble_size
        # )

        return model_ensemble

    @rule(
        target=ensembles,
        prior=ensembles,
    )
    def create_ensemble_from_prior(self, prior: Ensemble):
        assume(prior.uuid not in self.deleted_ensembles)
        prior_ensemble = self.storage.get_ensemble(prior.uuid)
        experiment_id = prior_ensemble.experiment_id
        size = prior_ensemble.ensemble_size
        ensemble = self.storage.create_ensemble(
            experiment_id, ensemble_size=size, prior_ensemble=prior.uuid
        )
        assert ensemble in self.storage.ensembles
        model_ensemble = Ensemble(ensemble.id)
        self.model[experiment_id].ensembles[ensemble.id] = model_ensemble
        # https://github.com/equinor/ert/issues/7046
        # assert (
        #    ensemble.get_ensemble_state()
        #    == [RealizationStorageState.PARENT_FAILURE] * size
        # )

        return model_ensemble

    @rule(model_experiment=experiments)
    def get_experiment(self, model_experiment: Experiment):
        storage_experiment = self.storage.get_experiment(model_experiment.uuid)
        assert storage_experiment.id == model_experiment.uuid
        assert sorted(model_experiment.ensembles) == sorted(
            e.id for e in storage_experiment.ensembles
        )
        assert (
            list(storage_experiment.response_configuration.values())
            == model_experiment.responses
        )
        assert model_experiment.observations == pytest.approx(
            storage_experiment.observations
        )

    @rule(model_ensemble=consumes(ensembles))
    def delete_ensemble(self, model_ensemble: Ensemble):
        assume(model_ensemble.uuid not in self.deleted_ensembles)
        ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        self.storage.delete_ensemble(ensemble)

        # The ensemble is no longer in storage
        with pytest.raises(KeyError):
            self.storage.get_ensemble(ensemble.id)

        # The "dangling" ensemble object cannot be used to load or save data
        with pytest.raises(ModeError):
            ensemble.load_all_gen_kw_data()
        with pytest.raises(ModeError):
            ensemble.save_parameters("param", 0, xr.Dataset({"values": [1, 2, 3]}))

        # The directory for the ensemble no longer exists
        assert not ensemble.path.exists()

        del self.model[ensemble.experiment_id].ensembles[ensemble.id]

    @rule(model_experiment=consumes(experiments))
    def delete_experiment(self, model_experiment: Experiment):
        experiment = self.storage.get_experiment(model_experiment.uuid)
        ensembles = list(experiment.ensembles)

        self.storage.delete_experiment(experiment)

        # The experiment is no longer in storage
        with pytest.raises(KeyError):
            self.storage.get_experiment(experiment.id)

        for ensemble in ensembles:
            self.deleted_ensembles[ensemble.id] = ensemble

        # The directories for the experiment nor its ensembles exists
        assert not experiment.path.exists()
        assert all(not x.path.exists() for x in ensembles)

        del self.model[experiment.id]

    @rule(model_ensemble=ensembles)
    def get_ensemble(self, model_ensemble: Ensemble):
        if model_ensemble.uuid not in self.deleted_ensembles:
            storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
            assert storage_ensemble.id == model_ensemble.uuid
        else:
            ensemble = self.deleted_ensembles[model_ensemble.uuid]
            with pytest.raises(KeyError):
                self.storage.get_ensemble(ensemble.id)

    @rule(model_ensemble=ensembles, data=st.data(), message=st.text())
    def set_failure(self, model_ensemble: Ensemble, data: st.DataObject, message: str):
        assume(model_ensemble.uuid not in self.deleted_ensembles)
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        assert storage_ensemble.id == model_ensemble.uuid

        realization = data.draw(
            st.integers(min_value=0, max_value=storage_ensemble.ensemble_size - 1)
        )

        storage_ensemble.set_failure(
            realization, RealizationStorageState.PARENT_FAILURE, message
        )
        model_ensemble.failure_messages[realization] = message

    @rule(model_ensemble=ensembles, data=st.data())
    def get_failure(self, model_ensemble: Ensemble, data: st.DataObject):
        assume(model_ensemble.uuid not in self.deleted_ensembles)
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        realization = data.draw(
            st.integers(min_value=0, max_value=storage_ensemble.ensemble_size - 1)
        )
        fail = self.storage.get_ensemble(model_ensemble.uuid).get_failure(realization)
        if realization in model_ensemble.failure_messages:
            assert fail is not None
            assert fail.message == model_ensemble.failure_messages[realization]
        else:
            assert fail is None or "Failure from prior" in fail.message

    def teardown(self):
        if self.storage is not None:
            self.storage.close()
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir)


TestStorage = StatefulTest.TestCase
