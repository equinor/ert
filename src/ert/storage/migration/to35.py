from __future__ import annotations

import logging
from pathlib import Path

from .migration_utils import experiment_index_files, read_json, write_json

logger = logging.getLogger(__name__)

info = "Move ES-MDA weights into experiment analysis_settings"


def _move_weights_into_analysis_settings(path: Path) -> None:
    for index_file in experiment_index_files(path):
        index_data = read_json(index_file)
        experiment_data = index_data.get("experiment", {})

        if "weights" not in experiment_data:
            continue

        weights = experiment_data.pop("weights")
        analysis_settings = experiment_data.setdefault("analysis_settings", {})
        analysis_settings.setdefault("weights", weights)

        write_json(index_file, index_data)
        logger.info("Moved weights into analysis_settings in %s", index_file)


def migrate(path: Path) -> None:
    _move_weights_into_analysis_settings(path)
