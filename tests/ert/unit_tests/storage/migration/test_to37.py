import json
from pathlib import Path

import pytest

from ert.storage.migration.to37 import migrate


def _write_index(ens_path: Path, index_data: dict) -> Path:
    ens_path.mkdir(parents=True)
    index_file = ens_path / "index.json"
    index_file.write_text(json.dumps(index_data), encoding="utf-8")
    return index_file


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        pytest.param(
            {
                "id": "ens-id",
                "ensemble": {"name": "batch_0", "iteration": 0, "is_improvement": True},
            },
            {"id": "ens-id", "ensemble": {"name": "batch_0", "iteration": 0}},
            id="removes_is_improvement",
        ),
        pytest.param(
            {"ensemble": {"name": "batch_1", "iteration": 1, "is_improvement": False}},
            {"ensemble": {"name": "batch_1", "iteration": 1}},
            id="removes_is_improvement_false",
        ),
        pytest.param(
            {"ensemble": {"name": "batch_5", "iteration": 3, "is_improvement": True}},
            {"ensemble": {"name": "batch_5", "iteration": 3}},
            id="removes_is_improvement_other_values",
        ),
        pytest.param(
            {"ensemble": {"name": "batch_0", "iteration": 0}},
            {"ensemble": {"name": "batch_0", "iteration": 0}},
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


def test_that_migration_does_not_fail_on_index_without_ensemble_entry(tmp_path):
    root = tmp_path / "project"
    index_file = _write_index(root / "ensembles" / "ensemble_1", {"id": "ens-id"})

    migrate(root)

    assert json.loads(index_file.read_text(encoding="utf-8")) == {"id": "ens-id"}
