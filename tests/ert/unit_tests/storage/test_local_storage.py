import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, PropertyMock, patch
from uuid import UUID

import hypothesis.strategies as st
import numpy as np
import polars
import pytest
import xarray as xr
from hypothesis import assume, given, note
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
from ert.config.gen_kw_config import TransformFunctionDefinition
from ert.config.general_observation import GenObservation
from ert.config.observation_vector import ObsVector
from ert.storage import ErtStorageException, open_storage
from ert.storage.local_storage import _LOCAL_STORAGE_VERSION
from ert.storage.mode import ModeError
from ert.storage.realization_storage_state import RealizationStorageState
from tests.ert.unit_tests.config.egrid_generator import egrids
from tests.ert.unit_tests.config.summary_generator import summaries, summary_variables


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


def test_that_loading_non_existing_experiment_throws(tmp_path):
    with open_storage(tmp_path, mode="w") as storage, pytest.raises(
        KeyError, match="Experiment with name 'non-existing-experiment' not found"
    ):
        storage.get_experiment_by_name("non-existing-experiment")


def test_that_loading_non_existing_ensemble_throws(tmp_path):
    with open_storage(tmp_path, mode="w") as storage, pytest.raises(
        KeyError, match="Ensemble with name 'non-existing-ensemble' not found"
    ):
        experiment = storage.create_experiment(name="test-experiment")
        experiment.get_ensemble_by_name("non-existing-ensemble")


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
            ensemble.save_response("RESPONSE", polars.DataFrame(), 0)

        # Test for dataset with 'values' but no actual data
        empty_data = polars.DataFrame(
            {
                "response_key": [],
                "report_step": [],
                "index": [],
                "values": [],
            }
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
        transform_function_definitions=[
            TransformFunctionDefinition("KEY1", "UNIFORM", [0, 1]),
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


@pytest.mark.timeout(10)
def test_open_storage_write_with_empty_directory(tmp_path, caplog):
    with open_storage(tmp_path / "storage", mode="w") as storage:
        _ = storage.create_experiment()
        assert len(list(storage.experiments)) == 1

    with open_storage(tmp_path / "storage", mode="w") as storage:
        _ = storage.create_experiment()
        assert len(list(storage.experiments)) == 1

    storage.refresh()
    assert len(list(storage.experiments)) == 0

    assert len(caplog.messages) == 0


def test_open_storage_read_with_empty_directory(tmp_path, caplog):
    with open_storage(tmp_path / "storage", mode="r") as storage:
        assert list(storage.ensembles) == []
        assert list(storage.experiments) == []


def test_open_storage_nested_dirs(tmp_path, caplog):
    with open_storage(tmp_path / "extra_level" / "storage", mode="w") as storage:
        assert storage.path.exists()


def test_open_storage_with_corrupted_storage(tmp_path):
    with open_storage(tmp_path / "storage", mode="w") as storage:
        storage.create_experiment().create_ensemble(name="prior", ensemble_size=1)
    os.remove(tmp_path / "storage" / "index.json")
    with pytest.raises(ErtStorageException, match="No index.json"):
        open_storage(tmp_path / "storage", mode="w")


def test_that_open_storage_in_read_mode_with_newer_version_throws_exception(
    tmp_path, caplog
):
    with open_storage(tmp_path, mode="w") as storage:
        storage._index.version = _LOCAL_STORAGE_VERSION + 1
        storage._save_index()

    with pytest.raises(
        ErtStorageException,
        match=f"Cannot open storage '{tmp_path}': Storage version {_LOCAL_STORAGE_VERSION+1} is newer than the current version {_LOCAL_STORAGE_VERSION}, upgrade ert to continue, or run with a different ENSPATH",
    ):
        open_storage(tmp_path, mode="r")


def test_that_open_storage_in_read_mode_with_older_version_throws_exception(
    tmp_path, caplog
):
    with open_storage(tmp_path, mode="w") as storage:
        storage._index.version = _LOCAL_STORAGE_VERSION - 1
        storage._save_index()

    with pytest.raises(
        ErtStorageException,
        match=f"Cannot open storage '{tmp_path}' in read-only mode: Storage version {_LOCAL_STORAGE_VERSION-1} is too old",
    ):
        open_storage(tmp_path, mode="r")


def test_that_open_storage_in_write_mode_with_newer_version_throws_exception(
    tmp_path, caplog
):
    with open_storage(tmp_path, mode="w") as storage:
        storage._index.version = _LOCAL_STORAGE_VERSION + 1
        storage._save_index()

    with pytest.raises(
        ErtStorageException,
        match=f"Cannot open storage '{tmp_path}': Storage version {_LOCAL_STORAGE_VERSION+1} is newer than the current version {_LOCAL_STORAGE_VERSION}, upgrade ert to continue, or run with a different ENSPATH",
    ):
        open_storage(tmp_path, mode="w")


def test_ensemble_no_parameters(storage):
    ensemble = storage.create_experiment(name="my-experiment").create_ensemble(
        ensemble_size=2,
        name="prior",
    )
    assert ensemble.get_ensemble_state() == [RealizationStorageState.HAS_DATA] * 2


def test_get_unique_experiment_name(snake_oil_storage):
    with patch(
        "ert.storage.local_storage.LocalStorage.experiments", new_callable=PropertyMock
    ) as experiments:
        # Its not possible to do MagicMock(name="experiment_name") therefore the workaround below
        names = [
            "experiment",
            "experiment_1",
            "experiment_8",
            "_d_e_",
            "___name__0___",
            "__name__1",
            "default",
        ]
        experiment_list = [MagicMock() for _ in range(len(names))]
        for k, v in zip(experiment_list, names):
            k.name = v
        experiments.return_value = experiment_list

        assert snake_oil_storage.get_unique_experiment_name("_d_e_") == "_d_e__0"
        assert (
            snake_oil_storage.get_unique_experiment_name("experiment") == "experiment_9"
        )
        assert (
            snake_oil_storage.get_unique_experiment_name("___name__0___")
            == "___name__0____0"
        )
        assert snake_oil_storage.get_unique_experiment_name("name") == "name"
        assert snake_oil_storage.get_unique_experiment_name("__name__") == "__name__"
        assert snake_oil_storage.get_unique_experiment_name("") == "default_0"


def add_to_name(prefix: str):
    def _inner(params):
        for param in params:
            param.name = prefix + param.name
        return params

    return _inner


parameter_configs = st.lists(
    st.one_of(
        st.builds(
            GenKwConfig,
            template_file=st.just(None),
            name=st.text(),
            output_file=st.just(None),
            update=st.booleans(),
            forward_init=st.booleans(),
            transform_function_definitions=st.just([]),
        ),
        st.builds(SurfaceConfig),
    ),
    unique_by=lambda x: x.name,
    min_size=1,
).map(add_to_name("parameter_"))

summary_selectors = st.one_of(
    summary_variables(), st.just("*"), summary_variables().map(lambda x: x + "*")
)

response_configs = st.lists(
    st.one_of(
        st.builds(
            GenDataConfig,
        ),
        st.builds(
            SummaryConfig,
            name=st.just("summary"),
            input_files=st.lists(
                st.text(
                    alphabet=st.characters(
                        min_codepoint=ord("A"), max_codepoint=ord("Z")
                    )
                ),
                min_size=1,
                max_size=1,
            ),
            keys=st.lists(summary_selectors, min_size=1),
        ),
    ),
    unique_by=lambda x: x.name,
    min_size=1,
)

ensemble_sizes = st.integers(min_value=1, max_value=1000)
coordinates = st.integers(min_value=1, max_value=100)

words = st.text(
    min_size=1,
    max_size=8,
    alphabet=st.characters(min_codepoint=ord("A"), max_codepoint=ord("Z")),
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
                min_size=1,
                max_size=1,
            ),
        ),
    ),
    obs_time=st.lists(
        st.datetimes(
            min_value=datetime.strptime("1969-1-1", "%Y-%m-%d"),
            max_value=datetime.strptime("3000-1-1", "%Y-%m-%d"),
        )
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


@pytest.mark.usefixtures("use_tmpdir")
@given(st.binary())
def test_write_transaction(data):
    with open_storage(".", "w") as storage:
        filepath = Path("./file.txt")
        storage._write_transaction(filepath, data)

        assert filepath.read_bytes() == data


class RaisingWriteNamedTemporaryFile:
    entered = False

    def __init__(self, *args, **kwargs):
        self.wrapped = tempfile.NamedTemporaryFile(*args, **kwargs)  # noqa
        RaisingWriteNamedTemporaryFile.entered = False

    def __enter__(self, *args, **kwargs):
        self.actual_handle = self.wrapped.__enter__(*args, **kwargs)
        mock_handle = MagicMock()
        RaisingWriteNamedTemporaryFile.entered = True

        def ctrlc(_):
            raise RuntimeError()

        mock_handle.write = ctrlc
        return mock_handle

    def __exit__(self, *args, **kwargs):
        self.wrapped.__exit__(*args, **kwargs)


def test_write_transaction_failure(tmp_path):
    with open_storage(tmp_path, "w") as storage:
        path = tmp_path / "file.txt"
        with patch(
            "ert.storage.local_storage.NamedTemporaryFile",
            RaisingWriteNamedTemporaryFile,
        ) as f, pytest.raises(RuntimeError):
            storage._write_transaction(path, b"deadbeaf")

        assert f.entered

        assert not path.exists()


def test_write_transaction_overwrites(tmp_path):
    with open_storage(tmp_path, "w") as storage:
        path = tmp_path / "file.txt"
        path.write_text("abc")
        storage._write_transaction(path, b"deadbeaf")
        assert path.read_bytes() == b"deadbeaf"


@dataclass
class Ensemble:
    uuid: UUID
    parameter_values: Dict[str, Any] = field(default_factory=dict)
    response_values: Dict[str, Any] = field(default_factory=dict)
    failure_messages: Dict[int, str] = field(default_factory=dict)


@dataclass
class Experiment:
    uuid: UUID
    ensembles: Dict[UUID, Ensemble] = field(default_factory=dict)
    parameters: List[ParameterConfig] = field(default_factory=list)
    responses: List[ResponseConfig] = field(default_factory=list)
    observations: Dict[str, polars.DataFrame] = field(default_factory=dict)


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
        self.tmpdir = tempfile.mkdtemp(prefix="StatefulStorageTest")
        self.storage = open_storage(self.tmpdir + "/storage/", "w")
        note(f"storage path is: {self.storage.path}")
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
    def create_field_list(self, fields):
        return fields

    @rule()
    def double_open_timeout(self):
        # Opening with write access will timeout when
        # already opened with mode="w" somewhere else
        with patch(
            "ert.storage.local_storage.LocalStorage.LOCK_TIMEOUT", 0.0
        ), pytest.raises(ErtStorageException):
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
        field_data=grid.flatmap(lambda g: arrays(np.float32, shape=g[1].shape)),
    )
    def write_error_in_save_field(self, model_ensemble: Ensemble, field_data):
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        parameters = model_ensemble.parameter_values.values()
        fields = [p for p in parameters if isinstance(p, Field)]
        iens = 0
        assume(not storage_ensemble.realizations_initialized([iens]))
        for f in fields:
            with patch(
                "ert.storage.local_storage.NamedTemporaryFile",
                RaisingWriteNamedTemporaryFile,
            ) as temp_file, pytest.raises(RuntimeError):
                storage_ensemble.save_parameters(
                    f.name,
                    iens,
                    xr.DataArray(
                        field_data,
                        name="values",
                        dims=["x", "y", "z"],  # type: ignore
                    ).to_dataset(),
                )

            assert temp_file.entered
        assert not storage_ensemble.realizations_initialized([iens])

    @rule(
        model_ensemble=ensembles,
    )
    def get_parameters(self, model_ensemble: Ensemble):
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        parameter_names = model_ensemble.parameter_values.keys()
        iens = 0

        for f in parameter_names:
            parameter_data = storage_ensemble.load_parameters(f, iens)
            xr.testing.assert_equal(
                model_ensemble.parameter_values[f],
                parameter_data["values"],
            )

    @rule(
        model_ensemble=ensembles,
        summary_data=summaries(
            start_date=st.datetimes(
                min_value=datetime.strptime("1969-1-1", "%Y-%m-%d"),
                max_value=datetime.strptime("2010-1-1", "%Y-%m-%d"),
            ),
            time_deltas=st.lists(
                st.floats(
                    min_value=0.1,
                    max_value=365,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                min_size=2,
                max_size=10,
            ),
        ),
    )
    def save_summary(self, model_ensemble: Ensemble, summary_data):
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        storage_experiment = storage_ensemble.experiment
        responses = storage_experiment.response_configuration.values()
        summary_configs = [p for p in responses if isinstance(p, SummaryConfig)]
        assume(summary_configs)
        summary = summary_configs[0]
        assume(summary.name not in model_ensemble.response_values)
        smspec, unsmry = summary_data
        smspec.to_file(self.tmpdir + f"/{summary.input_files[0]}.SMSPEC")
        unsmry.to_file(self.tmpdir + f"/{summary.input_files[0]}.UNSMRY")
        iens = 0

        try:
            ds = summary.read_from_file(self.tmpdir, iens)
        except ValueError as e:  # no match in keys
            assume(False)
            raise AssertionError() from e
        storage_ensemble.save_response(summary.response_type, ds, iens)

        model_ensemble.response_values[summary.name] = ds

    @rule(model_ensemble=ensembles)
    def get_responses(self, model_ensemble: Ensemble):
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        response_types = model_ensemble.response_values.keys()
        iens = 0

        for response_type in response_types:
            ensemble_data = storage_ensemble.load_responses(response_type, (iens,))
            model_data = model_ensemble.response_values[response_type]
            assert ensemble_data.equals(model_data)

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
        model_experiment = self.model[experiment_id]
        model_experiment.ensembles[ensemble.id] = model_ensemble
        state = [RealizationStorageState.PARENT_FAILURE] * size
        iens = 0
        if (
            list(prior.response_values.keys())
            == [r.name for r in model_experiment.responses]
            and iens not in prior.failure_messages
            and prior_ensemble.get_ensemble_state()[iens]
            != RealizationStorageState.PARENT_FAILURE
        ):
            state[iens] = RealizationStorageState.UNDEFINED
        assert ensemble.get_ensemble_state() == state

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
        for obskey, obs in model_experiment.observations.items():
            assert obskey in storage_experiment.observations
            assert obs.equals(storage_experiment.observations[obskey])

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

    @rule(model_ensemble=ensembles, data=st.data(), message=st.text())
    def write_error_in_set_failure(
        self,
        model_ensemble: Ensemble,
        data: st.DataObject,
        message: str,
    ):
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        realization = data.draw(
            st.integers(min_value=0, max_value=storage_ensemble.ensemble_size - 1)
        )
        assume(not storage_ensemble.has_failure(realization))

        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)

        with patch(
            "ert.storage.local_storage.NamedTemporaryFile",
            RaisingWriteNamedTemporaryFile,
        ) as f, pytest.raises(RuntimeError):
            storage_ensemble.set_failure(
                realization, RealizationStorageState.PARENT_FAILURE, message
            )
        assert f.entered

        assert not storage_ensemble.has_failure(realization)

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


TestStorage = pytest.mark.integration_test(StatefulStorageTest.TestCase)
