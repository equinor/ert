import json
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
from ert.storage import open_storage
from ert.storage.mode import ModeError
from ert.storage.realization_storage_state import RealizationStorageState
from tests.unit_tests.config.egrid_generator import egrids
from tests.unit_tests.config.summary_generator import summary_keys


def _ensembles(storage):
    return sorted(x.name for x in storage.ensembles)


def test_create_experiment(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment(name="test-experiment")

        experiment_path = Path(storage.path / "experiments" / str(experiment.id))
        assert experiment_path.exists()

        assert (experiment_path / experiment._parameter_file).exists()
        assert (experiment_path / experiment._responses_file).exists()

        with open(experiment_path / "index.json", encoding="utf-8", mode="r") as f:
            index = json.load(f)
            assert index["id"] == str(experiment.id)
            assert index["name"] == "test-experiment"


def test_that_saving_empty_responses_fails_nicely(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        ensemble = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        # Test for entirely empty dataset
        with pytest.raises(
            ValueError,
            match="Dataset for response group 'RESPONSE' must contain a 'values' variable",
        ):
            ensemble.save_response(
                "RESPONSE",
                xr.Dataset(),
                0,
            )

        # Test for dataset with 'values' but no actual data
        empty_data = xr.Dataset(
            {
                "values": (
                    ["report_step", "index"],
                    np.array([], dtype=float).reshape(0, 0),
                )
            },
            coords={
                "index": np.array([], dtype=int),
                "report_step": np.array([], dtype=int),
            },
        )
        with pytest.raises(
            ValueError,
            match="Responses RESPONSE are empty. Cannot proceed with saving to storage.",
        ):
            ensemble.save_response("RESPONSE", empty_data, 0)


def test_that_saving_empty_parameters_fails_nicely(tmp_path):
    with open_storage(tmp_path, mode="w") as storage:
        experiment = storage.create_experiment()
        prior = storage.create_ensemble(
            experiment, ensemble_size=1, iteration=0, name="prior"
        )

        # Test for entirely empty dataset
        with pytest.raises(
            ValueError,
            match="Dataset for parameter group 'PARAMETER' must contain a 'values' variable",
        ):
            prior.save_parameters("PARAMETER", 0, xr.Dataset())

        # Test for dataset with 'values' and 'transformed_values' but no actual data
        empty_data = xr.Dataset(
            {
                "values": ("names", np.array([], dtype=float)),
                "transformed_values": ("names", np.array([], dtype=float)),
                "names": (["names"], np.array([], dtype=str)),
            }
        )
        with pytest.raises(
            ValueError,
            match="Parameters PARAMETER are empty. Cannot proceed with saving to storage.",
        ):
            prior.save_parameters("PARAMETER", 0, empty_data)


def test_that_loading_parameter_via_response_api_fails(tmp_path):
    uniform_parameter = GenKwConfig(
        name="PARAMETER",
        forward_init=False,
        template_file="",
        transfer_function_definitions=[
            "KEY1 UNIFORM 0 1",
        ],
        output_file="kw.txt",
        update=True,
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
        assert _ensembles(storage) == []

    # Storage doesn't create an empty directory
    assert not (tmp_path / "empty").is_dir()


def test_open_empty_write(tmp_path):
    with open_storage(tmp_path / "empty", mode="w") as storage:
        assert _ensembles(storage) == []

    # Storage creates the directory
    assert (tmp_path / "empty").is_dir()


def test_refresh(tmp_path):
    with open_storage(tmp_path, mode="w") as accessor:
        experiment_id = accessor.create_experiment()
        with open_storage(tmp_path, mode="r") as reader:
            assert _ensembles(accessor) == _ensembles(reader)

            accessor.create_ensemble(experiment_id, name="foo", ensemble_size=42)
            # Reader does not know about the newly created ensemble
            assert _ensembles(accessor) != _ensembles(reader)

            reader.refresh()
            # Reader knows about it after the refresh
            assert _ensembles(accessor) == _ensembles(reader)


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
    min_size=1,
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
            keys=summary_keys,
            refcase=st.just(None),
        ),
    ),
    unique_by=lambda x: x.name,
    min_size=1,
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
            unique=True,
        ),
        std_scaling=arrays(np.double, shape=size),
    )
)


observations = st.builds(
    EnkfObs,
    obs_vectors=st.dictionaries(
        words,
        st.builds(
            ObsVector,
            observation_type=st.just(EnkfObservationImplementationType.GEN_OBS),
            observation_key=words,
            data_key=words,
            observations=st.dictionaries(
                st.integers(min_value=0, max_value=200),
                gen_observations,
                max_size=1,
                min_size=1,
            ),
        ),
    ),
)

small_ints = st.integers(min_value=1, max_value=10)


@st.composite
def fields(draw, egrid, num_fields=small_ints) -> List[Field]:
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
        for i in range(draw(num_fields))
    ]


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


class StatefulStorageTest(RuleBasedStateMachine):
    """
    This test runs several commands against storage and
    checks its return values against a simple key-value store
    (the model).

    see https://hypothesis.readthedocs.io/en/latest/stateful.html

    When the test fails, you get a printout like this:

    .. code-block:: text

        state = StatefulStorageTest()
        v1 = state.create_grid(egrid=EGrid(...))
        v2 = state.create_field_list(fields=[...])
        v3 = state.create_experiment(obs=EnkfObs(...), parameters=[...], responses=[...])
        v4 = state.create_ensemble(ensemble_size=1, model_experiment=v3)
        v5 = state.create_ensemble_from_prior(prior=v4)
        state.get_ensemble(model_ensemble=v5)
        state.teardown()

    This describes which rules are run (like create_experiment which corresponds to
    the same action storage api endpoint: self.storage.create_experiment), and which
    parameters are applied (e.g. v1 is in the grid bundle and is created by the rule
    state.create_grid).
    """

    def __init__(self):
        super().__init__()
        self.tmpdir = tempfile.mkdtemp()
        self.storage = open_storage(self.tmpdir + "/storage/", "w")
        self.model: Dict[UUID, Experiment] = {}
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
    def create_field_list(self, fields):  # noqa: PLR6301
        return fields

    @rule()
    def double_open_timeout(self):
        # Opening with write access will timeout when
        # already opened with mode="w" somewhere else
        with patch(
            "ert.storage.local_storage.LocalStorage.LOCK_TIMEOUT", 0.0
        ), pytest.raises(TimeoutError):
            open_storage(self.tmpdir + "/storage/", mode="w")

    @rule()
    def reopen(self):
        """
        closes as reopens the storage to ensure
        that doesn't effect its contents
        """
        ensembles = sorted(e.id for e in self.storage.ensembles)
        self.storage.close()
        self.storage = open_storage(self.tmpdir + "/storage/", mode="w")
        assert ensembles == sorted(e.id for e in self.storage.ensembles)

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

    @rule(
        model_ensemble=ensembles,
    )
    def get_field(self, model_ensemble: Ensemble):
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        field_names = model_ensemble.parameter_values.keys()
        for f in field_names:
            field_data = storage_ensemble.load_parameters(f, 1)
            np.testing.assert_array_equal(
                model_ensemble.parameter_values[f],
                field_data["values"],
            )

    @rule(model_ensemble=ensembles, parameter=words)
    def load_unknown_parameter(self, model_ensemble: Ensemble, parameter: str):
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

        assert (
            ensemble.get_ensemble_state()
            == [RealizationStorageState.UNDEFINED] * ensemble_size
        )
        assert np.all(np.logical_not(ensemble.get_realization_mask_with_responses()))

        return model_ensemble

    @rule(
        target=ensembles,
        prior=ensembles,
    )
    def create_ensemble_from_prior(self, prior: Ensemble):
        prior_ensemble = self.storage.get_ensemble(prior.uuid)
        experiment_id = prior_ensemble.experiment_id
        size = prior_ensemble.ensemble_size
        ensemble = self.storage.create_ensemble(
            experiment_id, ensemble_size=size, prior_ensemble=prior.uuid
        )
        assert ensemble in self.storage.ensembles
        model_ensemble = Ensemble(ensemble.id)
        self.model[experiment_id].ensembles[ensemble.id] = model_ensemble
        assert (
            ensemble.get_ensemble_state()
            == [RealizationStorageState.PARENT_FAILURE] * size
        )

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

    @rule(model_ensemble=ensembles)
    def get_ensemble(self, model_ensemble: Ensemble):
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        assert storage_ensemble.id == model_ensemble.uuid

    @rule(model_ensemble=ensembles, data=st.data(), message=st.text())
    def set_failure(self, model_ensemble: Ensemble, data: st.DataObject, message: str):
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


TestStorage = StatefulStorageTest.TestCase


def test_open_storage_write_with_empty_directory(tmp_path, caplog):
    with open_storage(tmp_path / "storage", mode="w") as storage:
        _ = storage.create_experiment()
        assert len(list(storage.experiments)) == 1

    assert len(caplog.messages) == 1
    assert "Unknown storage version in" in caplog.messages[0]

    caplog.clear()

    with open_storage(tmp_path / "storage", mode="w") as storage:
        _ = storage.create_experiment()
        assert len(list(storage.experiments)) == 1

    storage.refresh()
    assert len(list(storage.experiments)) == 0

    assert len(caplog.messages) == 0


def test_open_storage_read_with_empty_directory(tmp_path, caplog):
    with open_storage(tmp_path / "storage", mode="r"):
        assert len(caplog.messages) == 1
        assert "Unknown storage version in" in caplog.messages[0]
