import tempfile
import uuid
from pathlib import Path
from typing import List

import numpy as np
import py
import pytest
import xarray as xr
from flaky import flaky

from ert.analysis import smoother_update
from ert.config import EnkfObservationImplementationType, ErtConfig, SummaryConfig
from ert.enkf_main import EnKFMain
from ert.realization_state import RealizationState
from ert.storage import open_storage
from tests.performance_tests.performance_utils import make_poly_example


@pytest.fixture
def poly_template(monkeypatch):
    folder = py.path.local(tempfile.mkdtemp())
    script_path = Path(__file__).parent.resolve()
    folder = make_poly_example(
        folder,
        f"{script_path}/../../test-data/poly_template",
        gen_data_count=34,
        gen_data_entries=15,
        summary_data_entries=100,
        reals=2,
        summary_data_count=4000,
        sum_obs_count=450,
        gen_obs_count=34,
        sum_obs_every=10,
        gen_obs_every=1,
        parameter_entries=12,
        parameter_count=8,
        update_steps=1,
    )
    monkeypatch.chdir(folder)
    yield folder


@flaky(max_runs=5, min_passes=1)
@pytest.mark.limit_memory("130 MB")
@pytest.mark.integration_test
def test_memory_smoothing(poly_template):
    ert_config = ErtConfig.from_file("poly.ert")
    ert = EnKFMain(ert_config)
    fill_storage_with_data(poly_template, ert)
    with open_storage(poly_template / "ensembles", mode="w") as storage:
        prior_ens = storage.get_ensemble_by_name("prior")
        posterior_ens = storage.create_ensemble(
            prior_ens.experiment_id,
            ensemble_size=prior_ens.ensemble_size,
            iteration=1,
            name="posterior",
            prior_ensemble=prior_ens,
        )
        smoother_update(
            prior_ens,
            posterior_ens,
            str(uuid.uuid4()),
            ert.getLocalConfig(),
            ert.analysisConfig(),
        )


def fill_storage_with_data(poly_template: Path, ert: EnKFMain) -> None:
    path = Path(poly_template) / "ensembles"
    with open_storage(path, mode="w") as storage:
        ens_config = ert.ensembleConfig()
        experiment_id = storage.create_experiment(
            parameters=ens_config.parameter_configuration,
            observations=ert.getObservations().datasets,
        )
        source = storage.create_ensemble(experiment_id, name="prior", ensemble_size=100)
        observations = ert.getObservations()
        gen_obs_keys = observations.getTypedKeylist(
            EnkfObservationImplementationType.GEN_OBS
        )

        summary_obs_keys = ens_config.getKeylistFromImplType(SummaryConfig)
        realizations = list(range(ert.getEnsembleSize()))
        for real in realizations:
            for obs_key in gen_obs_keys:
                obs_vec = observations[obs_key]
                obs_highest_index_used = max(obs_vec.observations.keys())
                assert isinstance(obs_highest_index_used, int)
                source.save_response(
                    obs_vec.data_key, make_gen_data(obs_highest_index_used + 1), real
                )
            source.save_response(
                "summary",
                make_summary_data(summary_obs_keys, ens_config.refcase.numpy_dates),
                real,
            )
            source.state_map[real] = RealizationState.HAS_DATA

        ert.sample_prior(source, realizations, ens_config.parameters)

        storage.create_ensemble(
            source.experiment_id,
            ensemble_size=source.ensemble_size,
            iteration=1,
            name="target_ens",
            prior_ensemble=source,
        )


def make_gen_data(obs: int, min_val: float = 0, max_val: float = 5) -> xr.Dataset:
    data = np.random.default_rng().uniform(min_val, max_val, obs)
    return xr.Dataset(
        {"values": (["report_step", "index"], [data])},
        coords={"index": range(len(data)), "report_step": [0]},
    )


def make_summary_data(
    obs_keys: List[str],
    dates,
    min_val: float = 0,
    max_val: float = 5,
) -> xr.Dataset:
    data = np.random.default_rng().uniform(
        min_val, max_val, (len(obs_keys), len(dates))
    )
    return xr.Dataset(
        {"values": (["name", "time"], data)},
        coords={"time": dates, "name": obs_keys},
    )
