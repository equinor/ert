import json
from pathlib import Path

import pytest

from ert.storage.migration.to37 import migrate


def _write_index(ens_path: Path, index_data: dict) -> Path:
    ens_path.mkdir(parents=True)
    index_file = ens_path / "index.json"
    index_file.write_text(json.dumps(index_data), encoding="utf-8")
    return index_file


ENSEMBLE_ID = "6ec3d3f0-8b5b-4a3f-9c1a-3f6a3b0f1c11"
EXPERIMENT_ID = "1f2e3d4c-5b6a-4978-8899-aabbccddeeff"
PRIOR_ENSEMBLE_ID = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"


def _index_data(**overrides) -> dict:
    data = {
        "id": ENSEMBLE_ID,
        "experiment_id": EXPERIMENT_ID,
        "ensemble_size": 10,
        "iteration": 3,
        "name": "batch_0",
        "prior_ensemble_id": PRIOR_ENSEMBLE_ID,
        "started_at": "2023-01-01T00:00:00+00:00",
        "everest_realization_info": {
            "0": {"model_realization": 0, "perturbation": -1},
            "1": {"model_realization": 0, "perturbation": 0},
        },
    }
    data.update(overrides)
    return data


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        pytest.param(
            _index_data(is_improvement=True),
            _index_data(),
            id="removes_is_improvement_true",
        ),
        pytest.param(
            _index_data(is_improvement=False),
            _index_data(),
            id="removes_is_improvement_false",
        ),
        pytest.param(
            _index_data(is_improvement=None),
            _index_data(),
            id="removes_is_improvement_none",
        ),
        pytest.param(
            _index_data(),
            _index_data(),
            id="leaves_untouched_when_missing",
        ),
    ],
)
def test_that_migration_updates_ensemble_index(tmp_path, original, expected):
    root = tmp_path / "project"
    index_file = _write_index(root / "ensembles" / "ensemble_1", original)

    migrate(root)

    assert json.loads(index_file.read_text(encoding="utf-8")) == expected


def test_that_migration_does_not_fail_on_unexpectedly_structured_dirs(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    migrate(root)

    ensembles_dir = root / "ensembles"
    ensembles_dir.mkdir()
    migrate(root)

    not_an_ensemble = ensembles_dir / "not_a_directory.json"
    not_an_ensemble.write_text("{}", encoding="utf-8")
    migrate(root)

    ensemble_without_index_json = ensembles_dir / "ensemble_1"
    ensemble_without_index_json.mkdir()
    migrate(root)

    assert not_an_ensemble.read_text(encoding="utf-8") == "{}"
