import json
from pathlib import Path

from ert.storage.migration.to44 import migrate

EXPERIMENT_ID = "1f2e3d4c-5b6a-4978-8899-aabbccddeeff"
OTHER_EXPERIMENT_ID = "2a3b4c5d-6e7f-4a8b-9c0d-1e2f3a4b5c6d"
ENSEMBLE_ID = "6ec3d3f0-8b5b-4a3f-9c1a-3f6a3b0f1c11"
OTHER_ENSEMBLE_ID = "9a8b7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_ensemble(
    root: Path,
    ensemble_id: str,
    iteration: int = 0,
    experiment_id: str = EXPERIMENT_ID,
    started_at: str = "2026-01-01T00:00:00",
) -> None:
    _write_json(
        root / "ensembles" / ensemble_id / "index.json",
        {
            "id": ensemble_id,
            "experiment_id": experiment_id,
            "iteration": iteration,
            "started_at": started_at,
        },
    )


def _migrate_status_snapshot(root: Path, snapshot: dict) -> dict | None:
    experiment_dir = root / "experiments" / "exp1"
    _write_json(experiment_dir / "index.json", {"id": EXPERIMENT_ID})
    snapshot_path = (
        experiment_dir / "status" / f"iteration-{snapshot['iteration']}.json"
    )
    _write_json(snapshot_path, snapshot)

    migrate(root)

    if not snapshot_path.exists():
        return None
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def test_that_migration_adds_id_of_ensemble_at_the_snapshot_iteration(tmp_path):
    _write_ensemble(tmp_path, OTHER_ENSEMBLE_ID, iteration=0)
    _write_ensemble(tmp_path, ENSEMBLE_ID, iteration=10)

    migrated = _migrate_status_snapshot(tmp_path, {"iteration": 10})

    assert migrated == {"iteration": 10, "ensemble_id": ENSEMBLE_ID}


def test_that_migration_picks_latest_started_ensemble_when_iteration_is_shared(
    tmp_path,
):
    _write_ensemble(tmp_path, OTHER_ENSEMBLE_ID, started_at="2026-01-01T12:00:00")
    _write_ensemble(tmp_path, ENSEMBLE_ID, started_at="2026-01-02T00:00:00+00:00")

    migrated = _migrate_status_snapshot(tmp_path, {"iteration": 0})

    assert migrated == {"iteration": 0, "ensemble_id": ENSEMBLE_ID}


def test_that_migration_ignores_ensembles_from_other_experiments(tmp_path):
    _write_ensemble(tmp_path, ENSEMBLE_ID, started_at="2026-01-01T00:00:00")
    _write_ensemble(
        tmp_path,
        OTHER_ENSEMBLE_ID,
        experiment_id=OTHER_EXPERIMENT_ID,
        started_at="2026-02-01T00:00:00",
    )

    migrated = _migrate_status_snapshot(tmp_path, {"iteration": 0})

    assert migrated == {"iteration": 0, "ensemble_id": ENSEMBLE_ID}


def test_that_migration_removes_status_snapshot_without_matching_ensemble(tmp_path):
    _write_ensemble(tmp_path, ENSEMBLE_ID, iteration=0)

    assert _migrate_status_snapshot(tmp_path, {"iteration": 1}) is None


def test_that_migration_leaves_existing_ensemble_id_unchanged(tmp_path):
    _write_ensemble(tmp_path, ENSEMBLE_ID)
    snapshot = {"iteration": 0, "ensemble_id": OTHER_ENSEMBLE_ID}

    assert _migrate_status_snapshot(tmp_path, snapshot) == snapshot


def test_that_migration_accepts_storage_without_experiments(tmp_path):
    migrate(tmp_path)

    assert not (tmp_path / "experiments").exists()
