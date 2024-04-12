import tempfile
import uuid
from pathlib import Path
from typing import List

import numpy as np
import py
import pytest
import xarray as xr

from ert.analysis import smoother_update
from ert.config import ErtConfig
from ert.enkf_main import sample_prior
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


# @pytest.mark.flaky(reruns=5)
@pytest.mark.limit_memory("1 GB")
@pytest.mark.integration_test
def test_memory_smoothing(poly_template):
    ert_config = ErtConfig.from_file("poly.ert")
    fill_storage_with_data(poly_template, ert_config)
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
            list(ert_config.observation_keys),
            list(ert_config.ensemble_config.parameters),
        )


def fill_storage_with_data(poly_template: Path, ert_config: ErtConfig) -> None:
    path = Path(poly_template) / "ensembles"
    with open_storage(path, mode="w") as storage:
        ens_config = ert_config.ensemble_config
        experiment_id = storage.create_experiment(
            parameters=ens_config.parameter_configuration,
            responses=ens_config.response_configuration,
            observations=ert_config.observations.datasets,
        )
        source = storage.create_ensemble(experiment_id, name="prior", ensemble_size=100)

        realizations = list(range(ert_config.model_config.num_realizations))
        for obs_ds in ert_config.observations.datasets.values():
            response_type = obs_ds.attrs["response"]
            response_keys_for_observations = obs_ds["name"].data

            if response_type == "gen_data":
                for gen_data_key in response_keys_for_observations:
                    obs_highest_index_used = max(
                        obs_ds.sel(name=gen_data_key, drop=True).index.values
                    )
                    for real in realizations:
                        source.save_response(
                            gen_data_key,
                            make_gen_data(int(obs_highest_index_used) + 1),
                            real,
                        )

            if response_type == "summary":
                obs_time_list = ens_config.refcase.all_dates
                for real in realizations:
                    source.save_response(
                        "summary",
                        make_summary_data(
                            response_keys_for_observations, obs_time_list
                        ),
                        real,
                    )

        sample_prior(source, realizations, ens_config.parameters)

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
