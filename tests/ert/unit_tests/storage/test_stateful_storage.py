import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
import xarray as xr
from hypothesis import assume, note
from hypothesis.extra.numpy import arrays
from hypothesis.stateful import Bundle, RuleBasedStateMachine, initialize, rule
from resfo_utilities.testing import summaries, summary_variables

from ert.config import (
    Field,
    GenDataConfig,
    GenKwConfig,
    ParameterConfig,
    ResponseConfig,
    SummaryConfig,
    SurfaceConfig,
)
from ert.config._create_observation_dataframes import create_observation_dataframes
from ert.config._observations import (
    GeneralObservation,
)
from ert.config.distribution import DISTRIBUTION_CLASSES
from ert.field_utils import ErtboxParameters
from ert.storage import (
    ErtStorageException,
    RealizationStorageState,
    open_storage,
)
from tests.ert.grid_generator import xtgeo_box_grids
from tests.ert.unit_tests.storage._storage_test_helpers import (
    RaisingWriteNamedTemporaryFile,
)


def add_to_name(prefix: str):
    def _inner(params):
        for param in params:
            param.name = prefix + param.name
        return params

    return _inner


words = st.text(
    min_size=1,
    max_size=4,
    alphabet=st.characters(min_codepoint=ord("A"), max_codepoint=ord("Z")),
)


parameter_configs = st.lists(
    st.one_of(
        st.builds(
            GenKwConfig,
            name=words,
            group=words,
            update_strategy=st.sampled_from([None, "global", "adaptive", "distance"]),
            distribution=st.sampled_from(
                [clas.lower() for clas in DISTRIBUTION_CLASSES]
            ).map(lambda c: {"name": c}),
        ),
        st.builds(
            SurfaceConfig,
            name=words,
            ncol=st.integers(min_value=1, max_value=100),
            nrow=st.integers(min_value=1, max_value=100),
            xori=st.floats(allow_infinity=False, allow_nan=False),
            yori=st.floats(allow_infinity=False, allow_nan=False),
            xinc=st.floats(allow_infinity=False, allow_nan=False),
            yinc=st.floats(allow_infinity=False, allow_nan=False),
            rotation=st.floats(allow_infinity=False, allow_nan=False),
            yflip=st.sampled_from([-1, 1]),
        ),
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
            input_files=st.lists(
                st.text(
                    alphabet=st.characters(
                        min_codepoint=ord("A"), max_codepoint=ord("Z")
                    )
                ),
                min_size=1,
                max_size=1,
            ),
            keys=st.lists(summary_selectors),
        ),
    ),
    unique_by=lambda x: x.type,
    min_size=1,
)


ensemble_sizes = st.integers(min_value=1, max_value=1000)


general_observations = st.builds(
    GeneralObservation,
    name=words,
    data=words,
    restart=st.integers(min_value=1, max_value=10000),
    index=st.integers(min_value=1, max_value=10000),
    value=st.floats(allow_nan=False, allow_infinity=False),
    error=st.floats(allow_nan=False, allow_infinity=False, min_value=0.001),
)


observations = st.lists(
    general_observations,
    min_size=0,
    max_size=10,
)


small_ints = st.integers(min_value=1, max_value=10)


@st.composite
def fields(draw, egrid, num_fields=small_ints) -> list[Field]:
    grid_file, grid = egrid
    nx = grid.ncol
    ny = grid.nrow
    nz = grid.nlay
    return [
        draw(
            st.builds(
                Field,
                ertbox_params=st.builds(
                    ErtboxParameters, nx=st.just(nx), ny=st.just(ny), nz=st.just(nz)
                ),
                name=st.just(f"Field{i}"),
                file_format=st.just("roff_binary"),
                grid_file=st.just(grid_file),
                output_file=st.just(Path(f"field{i}.roff")),
            )
        )
        for i in range(draw(num_fields))
    ]


@dataclass
class Ensemble:
    uuid: UUID
    parameter_values: dict[str, Any] = field(default_factory=dict)
    response_values: dict[str, Any] = field(default_factory=dict)
    failure_messages: dict[int, str] = field(default_factory=dict)


@dataclass
class Experiment:
    uuid: UUID
    ensembles: dict[UUID, Ensemble] = field(default_factory=dict)
    parameters: list[ParameterConfig] = field(default_factory=list)
    responses: list[ResponseConfig] = field(default_factory=list)
    observations: dict[str, pl.DataFrame] = field(default_factory=dict)


class StatefulStorageTest(RuleBasedStateMachine):
    """
    This test runs several commands against storage and
    checks its return values against a simple key-value store
    (the model).

    see https://hypothesis.readthe@docs.io/en/latest/stateful.html

    When the test fails, you get a printout like this:

    .. code-block:: text

        state = StatefulStorageTest()
        v1 = state.create_grid(egrid=EGrid(...))
        v2 = state.create_field_list(fields=[...])
        v3 = state.create_experiment(obs=(...), parameters=[...], responses=[...])
        v4 = state.create_ensemble(ensemble_size=1, model_experiment=v3)
        v5 = state.create_ensemble_from_prior(prior=v4)
        state.get_ensemble(model_ensemble=v5)
        state.teardown()

    This describes which rules are run (like create_experiment which corresponds to
    the same action storage api endpoint: self.storage.create_experiment), and which
    parameters are applied (e.g. v1 is in the grid bundle and is created by the rule
    state.create_grid).
    """

    def __init__(self) -> None:
        super().__init__()
        self.tmpdir = tempfile.mkdtemp(prefix="StatefulStorageTest")
        self.storage = open_storage(self.tmpdir + "/storage/", mode="w")
        note(f"storage path is: {self.storage.path}")
        self.model: dict[UUID, Experiment] = {}
        assert list(self.storage.ensembles) == []

        # Realization to save/delete params/responses
        # (all other reals are not modified throughout every run of this test)
        self.iens_to_edit = 0

    experiments = Bundle("experiments")
    ensembles = Bundle("ensembles")
    ensembles_with_parameters = Bundle("ensembles_with_parameters")
    field_list = Bundle("field_list")
    grid = Bundle("grid")

    @initialize(target=grid, egrid=xtgeo_box_grids())
    def create_grid(self, egrid):
        grid_file = self.tmpdir + "/grid.egrid"
        egrid.to_file(grid_file, fformat="egrid")
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
        with (
            patch("ert.storage.local_storage.LocalStorage.LOCK_TIMEOUT", 0.0),
            pytest.raises(ErtStorageException),
        ):
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
        parameters: list[ParameterConfig],
        responses: list[ResponseConfig],
        obs,
    ):
        experiment_id = self.storage.create_experiment(
            experiment_config={
                "parameter_configuration": [
                    p.model_dump(mode="json") for p in parameters
                ],
                "response_configuration": [
                    r.model_dump(mode="json") for r in responses
                ],
                "observations": [o.model_dump(mode="json") for o in obs],
            }
        ).id
        model_experiment = Experiment(experiment_id)
        model_experiment.parameters = parameters
        model_experiment.responses = responses
        model_experiment.observations = create_observation_dataframes(obs)

        # Ensure that there is at least one ensemble in the experiment
        # to avoid https://github.com/equinor/ert/issues/7040
        ensemble = self.storage.create_ensemble(experiment_id, ensemble_size=1)
        model_experiment.ensembles[ensemble.id] = Ensemble(ensemble.id)

        self.model[model_experiment.uuid] = model_experiment

        return model_experiment

    @rule(
        model_ensemble=ensembles,
        data=st.data(),
        grid=grid,
        target=ensembles_with_parameters,
    )
    def save_parameters(self, model_ensemble: Ensemble, grid, data):
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        parameters = storage_ensemble.experiment.parameter_configuration

        # Ensembles w/ parent failure will never have parameters written to them
        assume(
            storage_ensemble.get_ensemble_state()[self.iens_to_edit]
            != RealizationStorageState.FAILURE_IN_PARENT
        )

        for p in parameters.values():
            match p:
                case Field():
                    field_data = xr.DataArray(
                        data.draw(
                            arrays(
                                np.float32,
                                shape=(grid[1].ncol, grid[1].nrow, grid[1].nlay),
                            )
                        ),
                        name="values",
                        dims=["x", "y", "z"],  # type: ignore
                    ).to_dataset()

                    model_ensemble.parameter_values[p.name] = field_data
                    storage_ensemble.save_parameters(
                        field_data,
                        p.name,
                        self.iens_to_edit,
                    )
                case GenKwConfig():
                    scalar_value = data.draw(
                        st.floats(allow_infinity=False, allow_nan=False)
                    )
                    model_ensemble.parameter_values[p.name] = scalar_value
                    storage_ensemble.save_parameters(
                        pl.DataFrame(
                            {p.name: [scalar_value]},
                            schema={p.name: pl.Float64},
                        ).with_columns(pl.Series("realization", [self.iens_to_edit]))
                    )
                case SurfaceConfig():
                    surface_data = xr.DataArray(
                        data.draw(arrays(np.float32, shape=(p.ncol, p.nrow))),
                        name="values",
                        dims=["x", "y"],  # type: ignore
                    ).to_dataset()
                    model_ensemble.parameter_values[p.name] = surface_data
                    storage_ensemble.save_parameters(
                        surface_data, p.name, self.iens_to_edit
                    )
        return model_ensemble

    @rule(
        model_ensemble=ensembles,
        field_data=grid.flatmap(
            lambda g: arrays(np.float32, shape=(g[1].ncol, g[1].nrow, g[1].nlay))
        ),
    )
    def write_error_in_save_field(self, model_ensemble: Ensemble, field_data):
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)

        # Ensembles w/ parent failure will never have parameters written to them
        assume(
            storage_ensemble.get_ensemble_state()[self.iens_to_edit]
            != RealizationStorageState.FAILURE_IN_PARENT
        )

        parameters = model_ensemble.parameter_values.values()
        fields = [p for p in parameters if isinstance(p, Field)]

        assume(
            not storage_ensemble.get_realization_mask_with_parameters()[
                self.iens_to_edit
            ]
        )
        for f in fields:
            with (
                patch(
                    "ert.storage.local_storage.NamedTemporaryFile",
                    RaisingWriteNamedTemporaryFile,
                ) as temp_file,
                pytest.raises(RuntimeError),
            ):
                storage_ensemble.save_parameters(
                    xr.DataArray(
                        field_data,
                        name="values",
                        dims=["x", "y", "z"],  # type: ignore
                    ).to_dataset(),
                    f.name,
                    self.iens_to_edit,
                )

            assert temp_file.entered
        assert not storage_ensemble.get_realization_mask_with_parameters()[
            self.iens_to_edit
        ]

    @rule(model_ensemble=ensembles_with_parameters, transformed=st.booleans())
    def get_parameters(self, model_ensemble: Ensemble, transformed: bool):
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        parameters = storage_ensemble.experiment.parameter_configuration

        for name, config in parameters.items():
            if name in model_ensemble.parameter_values:
                parameter_data = storage_ensemble.load_parameters(
                    name, self.iens_to_edit, transformed=transformed
                )
                match config:
                    case Field():
                        xr.testing.assert_equal(
                            model_ensemble.parameter_values[name],
                            parameter_data,
                        )
                    case GenKwConfig():
                        parameter_value = model_ensemble.parameter_values[name]
                        if transformed:
                            parameter_value = config.transform_series(
                                pl.Series(
                                    [parameter_value],
                                    dtype=pl.Float64,
                                )
                            )[0]
                        assert (
                            np.isnan(parameter_value)
                            and np.isnan(parameter_data[name][0])
                        ) or parameter_value == parameter_data[name][0]
                    case SurfaceConfig():
                        xr.testing.assert_equal(
                            model_ensemble.parameter_values[name],
                            parameter_data,
                        )

    @rule(model_ensemble=ensembles_with_parameters, transformed=st.booleans())
    def get_parameter_group(self, model_ensemble: Ensemble, transformed: bool):
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        parameters = storage_ensemble.experiment.parameter_configuration
        groups = {
            p.group_name
            for p in parameters.values()
            if p.name != p.group_name and p.group_name
        }
        if not groups:
            return

        for name, config in parameters.items():
            if name in model_ensemble.parameter_values and config.group_name in groups:
                group_data = storage_ensemble.load_parameters(
                    config.group_name, self.iens_to_edit, transformed=transformed
                )
                # Only GenKwConfig is grouped by anything other than name
                assert isinstance(config, GenKwConfig)
                parameter_value = model_ensemble.parameter_values[name]
                if transformed:
                    parameter_value = config.transform_series(
                        pl.Series(
                            [parameter_value],
                            dtype=pl.Float64,
                        )
                    )[0]
                assert (
                    np.isnan(parameter_value) and np.isnan(group_data[name][0])
                ) or parameter_value == group_data[name][0]

    @rule(model_ensemble=ensembles, data=st.data())
    def save_summary(self, model_ensemble: Ensemble, data):
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        storage_experiment = storage_ensemble.experiment

        assume(
            storage_ensemble.get_ensemble_state()[self.iens_to_edit]
            != RealizationStorageState.FAILURE_IN_PARENT
        )

        # Enforce the summary data to respect the
        # scheme outlined in the response configs
        smry_config = storage_experiment.response_configuration.get("summary")

        if not smry_config:
            assume(False)
            raise AssertionError

        expected_summary_keys = (
            st.just(smry_config.keys)
            if smry_config.has_finalized_keys
            else st.lists(summary_variables())
        ).map(lambda xs: [x for x in xs if x != "TIME"])

        summaries_strategy = summaries(
            summary_keys=expected_summary_keys,
            start_date=st.datetimes(
                min_value=datetime.strptime("1969-1-1", "%Y-%m-%d"),  # noqa: DTZ007
                max_value=datetime.strptime("2010-1-1", "%Y-%m-%d"),  # noqa: DTZ007
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
        )
        summary_data = data.draw(summaries_strategy)

        responses = storage_experiment.response_configuration.values()
        summary_configs = [p for p in responses if isinstance(p, SummaryConfig)]
        assume(summary_configs)
        summary = summary_configs[0]
        assume(summary.type not in model_ensemble.response_values)
        smspec, unsmry = summary_data
        smspec.to_file(self.tmpdir + f"/{summary.input_files[0]}.SMSPEC")
        unsmry.to_file(self.tmpdir + f"/{summary.input_files[0]}.UNSMRY")

        ds = summary.read_from_file(self.tmpdir, self.iens_to_edit, 0)
        storage_ensemble.save_response(summary.type, ds, self.iens_to_edit)

        model_ensemble.response_values[summary.type] = ds

        model_experiment = self.model[storage_experiment.id]
        response_keys = set(ds["response_key"].unique())

        model_smry_config = next(
            config for config in model_experiment.responses if config.type == "summary"
        )

        if not model_smry_config.has_finalized_keys:
            model_smry_config.keys = sorted(response_keys)
            model_smry_config.has_finalized_keys = True

    @rule(model_ensemble=ensembles)
    def get_responses(self, model_ensemble: Ensemble):
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        response_types = model_ensemble.response_values.keys()

        for response_type in response_types:
            ensemble_data = storage_ensemble.load_responses(
                response_type, (self.iens_to_edit,)
            )
            model_data = model_ensemble.response_values[response_type]
            assert ensemble_data.equals(model_data)

    @rule(model_ensemble=ensembles, parameter=words)
    def load_unknown_parameter(self, model_ensemble: Ensemble, parameter: str):
        storage_ensemble = self.storage.get_ensemble(model_ensemble.uuid)
        experiment_id = storage_ensemble.experiment_id
        parameter_names = {p.name for p in self.model[experiment_id].parameters}
        group_names = {p.group_name for p in self.model[experiment_id].parameters}
        assume(parameter not in parameter_names and parameter not in group_names)
        with pytest.raises(
            KeyError,
            match=f"{parameter} is not registered|No SCALAR dataset in",
        ):
            _ = storage_ensemble.load_parameters(parameter, self.iens_to_edit)

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

        is_expecting_responses = any(
            len(config.keys) for config in model_experiment.responses
        )

        if is_expecting_responses:
            assert all(
                (RealizationStorageState.UNDEFINED in s)
                for s in ensemble.get_ensemble_state()
            )
            assert np.all(
                np.logical_not(ensemble.get_realization_mask_with_responses())
            )
        else:
            assert all(
                RealizationStorageState.RESPONSES_LOADED in state
                for state in ensemble.get_ensemble_state()
            )
            assert np.all(ensemble.get_realization_mask_with_responses())

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

        prior_state = prior_ensemble.get_ensemble_state()
        edited_prior_state = prior_state[self.iens_to_edit]

        posterior_state = ensemble.get_ensemble_state()
        edited_posterior_state = posterior_state[self.iens_to_edit]

        if edited_prior_state.intersection(
            {
                RealizationStorageState.UNDEFINED,
                RealizationStorageState.FAILURE_IN_PARENT,
                RealizationStorageState.FAILURE_IN_CURRENT,
            }
        ):
            assert RealizationStorageState.FAILURE_IN_PARENT in edited_posterior_state
        else:
            is_expecting_responses = (
                sum(len(config.keys) for config in model_experiment.responses) > 0
            )
            # If expecting no responses, i.e., it has empty .keys in all response
            # configs, it will be a HAS_DATA even if no responses were ever saved
            if not is_expecting_responses:
                assert (
                    RealizationStorageState.RESPONSES_LOADED in edited_posterior_state
                )
            else:
                assert RealizationStorageState.UNDEFINED in edited_posterior_state

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
            realization, RealizationStorageState.FAILURE_IN_PARENT, message
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

        with (
            patch(
                "ert.storage.local_storage.NamedTemporaryFile",
                RaisingWriteNamedTemporaryFile,
            ) as f,
            pytest.raises(RuntimeError),
        ):
            storage_ensemble.set_failure(
                realization, RealizationStorageState.FAILURE_IN_PARENT, message
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


TestStorage = pytest.mark.fuzzing(pytest.mark.slow(StatefulStorageTest.TestCase))
