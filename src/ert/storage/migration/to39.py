from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .migration_utils import experiment_index_files, read_json, write_json

logger = logging.getLogger(__name__)
info = """
Remove redundant 'restart_run' Multiple Data Assimilation class key.
Convert emtpy 'prior_ensemble_id' to None.
"""


def _process_mda_experiments(
    path: Path, callback: Callable[[dict[str, Any]], None]
) -> None:
    """Apply a transformation to all MDA experiment index files."""
    for index_file in experiment_index_files(path):
        index_data = read_json(index_file)
        experiment_data = index_data.get("experiment", {})

        if experiment_data.get("experiment_type") != "Multiple Data Assimilation":
            continue

        callback(experiment_data)
        write_json(index_file, index_data)


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
