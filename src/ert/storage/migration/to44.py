from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

info = "Add ensemble id to status snapshots"


def _parse_started_at(value: str) -> datetime:
    started_at = datetime.fromisoformat(value)
    if started_at.tzinfo is None:
        return started_at.replace(tzinfo=UTC)
    return started_at


def _latest_ensemble_ids_by_experiment_and_iteration(
    ensembles_dir: Path,
) -> dict[tuple[str, int], str]:
    started_at_and_id: dict[tuple[str, int], tuple[datetime, str]] = {}
    for index_file in ensembles_dir.glob("*/index.json"):
        index = json.loads(index_file.read_text(encoding="utf-8"))
        key = (index["experiment_id"], index["iteration"])
        candidate = (_parse_started_at(index["started_at"]), index["id"])
        if key not in started_at_and_id or candidate > started_at_and_id[key]:
            started_at_and_id[key] = candidate
    return {key: ensemble_id for key, (_, ensemble_id) in started_at_and_id.items()}


def _add_ensemble_id_to_status_snapshots(path: Path) -> None:
    experiments_dir = path / "experiments"
    if not experiments_dir.exists():
        return

    ensemble_ids = _latest_ensemble_ids_by_experiment_and_iteration(path / "ensembles")

    for index_file in experiments_dir.glob("*/index.json"):
        experiment_id = json.loads(index_file.read_text(encoding="utf-8"))["id"]
        for snapshot_file in (index_file.parent / "status").glob("iteration-*.json"):
            snapshot = json.loads(snapshot_file.read_text(encoding="utf-8"))
            if "ensemble_id" in snapshot:
                continue

            ensemble_id = ensemble_ids.get((experiment_id, snapshot["iteration"]))
            if ensemble_id is None:
                snapshot_file.unlink()
                logger.info(
                    "Removed status snapshot without matching ensemble: %s",
                    snapshot_file,
                )
                continue

            snapshot["ensemble_id"] = ensemble_id
            snapshot_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


def migrate(path: Path) -> None:
    _add_ensemble_id_to_status_snapshots(path)
