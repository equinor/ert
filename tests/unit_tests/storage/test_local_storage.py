import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List
from unittest.mock import patch
from uuid import UUID

import hypothesis.strategies as st
import pytest
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule

from ert.config import (
    GenDataConfig,
    GenKwConfig,
    ParameterConfig,
    ResponseConfig,
    SummaryConfig,
    SurfaceConfig,
)
from ert.storage import StorageReader, open_storage
from ert.storage import local_storage as local
from ert.storage.realization_storage_state import RealizationStorageState
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


@st.composite
def refcase(draw):
    datetimes = draw(st.lists(st.datetimes()))
    container_type = draw(st.sampled_from([set(), list(), None]))
    if isinstance(container_type, set):
        return set(datetimes)
    elif isinstance(container_type, list):
        return [str(date) for date in datetimes]
    return None


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
            keys=st.lists(summary_keys),
            refcase=refcase(),
        ),
    ),
    unique_by=lambda x: x.name,
)

ensemble_sizes = st.integers(min_value=1, max_value=1000)


@dataclass
class Experiment:
    ensembles: List[UUID] = field(default_factory=list)
    parameters: List[ParameterConfig] = field(default_factory=list)
    responses: List[ResponseConfig] = field(default_factory=list)


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
        parameters=parameter_configs,
        responses=response_configs,
    )
    def create_experiment(
        self, parameters: List[ParameterConfig], responses: List[ResponseConfig]
    ):
        experiment_id = self.storage.create_experiment(
            parameters=parameters, responses=responses
        ).id
        self.experiments[experiment_id].parameters = parameters
        self.experiments[experiment_id].responses = responses

        # Ensure that there is at least one ensemble in the experiment
        # to avoid https://github.com/equinor/ert/issues/7040
        ensemble = self.storage.create_ensemble(experiment_id, ensemble_size=1)
        self.experiments[experiment_id].ensembles.append(ensemble.id)

        return experiment_id

    @rule(
        target=ensemble_ids,
        experiment=experiment_ids,
        ensemble_size=ensemble_sizes,
    )
    def create_ensemble(self, experiment: UUID, ensemble_size: int):
        ensemble = self.storage.create_ensemble(experiment, ensemble_size=ensemble_size)
        assert ensemble in self.storage.ensembles
        self.experiments[experiment].ensembles.append(ensemble.id)

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
        self.experiments[experiment].ensembles.append(ensemble.id)
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
