from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
info = """
Remove redundant 'restart_run' Multiple Data Assimilation class key.
Convert emtpy 'prior_ensemble_id' to None.
"""


def _process_mda_experiments(
    path: Path, callback: Callable[[dict[str, Any]], None]
) -> None:
    """Apply a transformation to all MDA experiment index files."""
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

        if experiment_data.get("experiment_type") != "Multiple Data Assimilation":
            continue

        callback(experiment_data)
        index_file.write_text(json.dumps(index_data, indent=2), encoding="utf-8")


def migrate(path: Path) -> None:
    def remove_restart_run(exp_data: dict[str, Any]) -> None:
        exp_data.pop("restart_run", None)

    def fix_prior_ensemble_id(exp_data: dict[str, Any]) -> None:
        if "prior_ensemble_id" not in exp_data:
            return
        if not exp_data.get("prior_ensemble_id"):
            exp_data["prior_ensemble_id"] = None

    _process_mda_experiments(path, remove_restart_run)
    _process_mda_experiments(path, fix_prior_ensemble_id)
