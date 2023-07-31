import tempfile
import uuid
from pathlib import Path
from typing import List
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import py
import pytest
import xarray as xr

from ert._c_wrappers.enkf import EnKFMain, EnkfObservationImplementationType, ErtConfig
from ert.analysis import ESUpdate
from ert.config import SummaryConfig
from ert.realization_state import RealizationState
from ert.storage import EnsembleAccessor, EnsembleReader
from ert.storage.local_ensemble import LocalEnsembleAccessor
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


@pytest.mark.limit_memory("130 MB")
@pytest.mark.integration_test
@pytest.mark.xfail(reason="Memory usage is variable")
def test_memory_smoothing(poly_template):
    ert_config = ErtConfig.from_file("poly.ert")
    ert = EnKFMain(ert_config)
    tgt = mock_target_accessor()
    src = make_source_accessor(poly_template, ert)
    smoother = ESUpdate(ert)
    smoother.smootherUpdate(src, tgt, str(uuid.uuid4()))


def mock_target_accessor() -> EnsembleAccessor:
    return Mock(spec=EnsembleAccessor)


def make_source_accessor(path: Path, ert: EnKFMain) -> EnsembleReader:
    realisations = ert.getEnsembleSize()
    uuid = uuid4()
    path = Path(path) / "ensembles" / str(uuid)
    source = LocalEnsembleAccessor.create(
        None,
        path,
        uuid,
        ensemble_size=realisations,
        experiment_id=uuid4(),
        iteration=1,
        name="default",
        prior_ensemble_id=None,
        refcase=None,
    )

    ens_config = ert.ensembleConfig()
    obs_keys = ert.getObservations().getTypedKeylist(
        EnkfObservationImplementationType.GEN_OBS
    )
    observations = ert.getObservations()

    obs_data_keys = ens_config.getKeylistFromImplType(SummaryConfig)
    for real in range(realisations):
        for obs_key in obs_keys:
            obs = observations[obs_key]
            obs_vec = obs.observations[0]  # Ignores all other time points for now
            obs_highest_index_used = obs_vec.getDataIndex(len(obs_vec) - 1)
            source.save_response(
                obs.getDataKey(), make_gen_data(obs_highest_index_used + 1), real
            )
        source.save_response(
            "summary",
            make_summary_data(obs_data_keys, ens_config.refcase.numpy_dates),
            real,
        )
        source.state_map[real] = RealizationState.HAS_DATA

    ert.sample_prior(source, list(range(realisations)), ens_config.parameters)

    return source


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
