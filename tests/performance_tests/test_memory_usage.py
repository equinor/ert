import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import Mock
from uuid import uuid4

import numpy as np
import py
import pytest

from ert._c_wrappers.enkf import (
    EnKFMain,
    EnkfObservationImplementationType,
    ErtConfig,
    ErtImplType,
    RealizationStateEnum,
)
from ert.analysis import ESUpdate
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
def test_memory_smoothing(poly_template):
    ert_config = ErtConfig.from_file("poly.ert")
    ert = EnKFMain(ert_config)
    tgt = mock_target_accessor()
    src = make_source_accessor(poly_template, ert)
    smoother = ESUpdate(ert)
    smoother.smootherUpdate(src, tgt, str(uuid.uuid4()))


def mock_target_accessor() -> EnsembleAccessor:
    target = Mock(spec=EnsembleAccessor)
    target.field_has_info.return_value = True
    return target


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
    gen_data_input = []
    for obs_key in obs_keys:
        obs = observations[obs_key]
        obs_vec = obs.getNode(0)  # Ignores all other time points for now
        obs_highest_index_used = obs_vec.getDataIndex(len(obs_vec) - 1)
        gen_data_input.append((obs.getDataKey(), obs_highest_index_used + 1))

    obs_data_keys = ens_config.getKeylistFromImplType(ErtImplType.SUMMARY)
    for real in range(realisations):
        source.save_gen_data(make_gen_data(gen_data_input), real)
        source.save_summary_data(
            make_summary_data(len(obs_data_keys), len(ens_config.refcase.numpy_dates)),
            obs_data_keys,
            ens_config.refcase.numpy_dates,
            real,
        )
        source.state_map[real] = RealizationStateEnum.STATE_HAS_DATA

    ert.sample_prior(source, list(range(realisations)), ens_config.parameters)

    return source


def make_gen_data(
    observation_list: List[Tuple[str, int]], min_val: float = 0, max_val: float = 5
) -> Dict[str, List[float]]:
    gen_data: Dict[str, List[float]] = {}
    for obs in observation_list:
        gen_data[f"{obs[0]}@0"] = list(
            np.random.default_rng().uniform(min_val, max_val, obs[1])
        )
    return gen_data


def make_summary_data(
    summary_data_count: int,
    summary_data_entries: int,
    min_val: float = 0,
    max_val: float = 5,
):
    return np.random.default_rng().uniform(
        min_val, max_val, (summary_data_count, summary_data_entries)
    )
