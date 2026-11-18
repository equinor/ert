from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import cast

logger = logging.getLogger(__name__)

info = (
    "Add ensemble id to status snapshots lacking it, using the single ensemble "
    "with the same experiment and iteration. Snapshots that are unreadable, "
    "or that match no ensemble or several ensembles, are removed"
)


def _ensemble_ids_by_experiment_and_iteration(
    ensembles_dir: Path,
) -> dict[tuple[str, int], list[str]]:
    ensemble_ids: dict[tuple[str, int], list[str]] = defaultdict(list)
    for index_file in ensembles_dir.glob("*/index.json"):
        index = json.loads(index_file.read_text(encoding="utf-8"))
        ensemble_ids[index["experiment_id"], index["iteration"]].append(index["id"])
    return ensemble_ids


def _read_status_snapshot(snapshot_file: Path) -> dict[str, object] | None:
    try:
        snapshot = json.loads(snapshot_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("iteration"), int):
        return None
    return snapshot


def _add_ensemble_id_to_status_snapshots(path: Path) -> None:
    experiments_dir = path / "experiments"
    if not experiments_dir.exists():
        return

    ensemble_ids = _ensemble_ids_by_experiment_and_iteration(path / "ensembles")

    for index_file in experiments_dir.glob("*/index.json"):
        experiment_id = json.loads(index_file.read_text(encoding="utf-8"))["id"]
        for snapshot_file in (index_file.parent / "status").glob("iteration-*.json"):
            snapshot = _read_status_snapshot(snapshot_file)
            if snapshot is None:
                snapshot_file.unlink()
                logger.warning("Removed unreadable status snapshot: %s", snapshot_file)
                continue
            if "ensemble_id" in snapshot:
                continue

            matching_ids = ensemble_ids.get(
                (experiment_id, cast(int, snapshot["iteration"])), []
            )
            if len(matching_ids) != 1:
                snapshot_file.unlink()
                logger.info(
                    "Removed status snapshot matching %d ensembles: %s",
                    len(matching_ids),
                    snapshot_file,
                )
                continue

            snapshot["ensemble_id"] = matching_ids[0]
            tmp_file = snapshot_file.with_suffix(".json.tmp")
            tmp_file.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
            tmp_file.replace(snapshot_file)


def migrate(path: Path) -> None:
    _add_ensemble_id_to_status_snapshots(path)
