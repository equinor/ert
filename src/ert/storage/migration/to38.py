from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

info = "Rename 'restart_run' Multiple Data Assimilation class key to 'select_prior'"


def _rename_restart_run_mda_key_to_select_prior(path: Path) -> None:
    experiments_dir = path / "experiments"
    if not experiments_dir.exists():
        return

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        index_file = exp_dir / "index.json"
        if not index_file.exists():
            continue

        index_data = json.loads(index_file.read_text(encoding="utf-8"))
        experiment_data = index_data.get("experiment", {})

        experiment_type = experiment_data.get("experiment_type", "")
        if experiment_type != "Multiple Data Assimilation":
            continue

        experiment_data["select_prior"] = experiment_data.pop("restart_run")

        index_file.write_text(json.dumps(index_data, indent=2), encoding="utf-8")


def migrate(path: Path) -> None:
    _rename_restart_run_mda_key_to_select_prior(path)
