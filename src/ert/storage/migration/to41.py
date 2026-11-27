from __future__ import annotations

import logging
from pathlib import Path

from .migration_utils import experiment_index_files, read_json, write_json

logger = logging.getLogger(__name__)

info = "Add offsets to the Everest objectives response configuration"


def _add_objective_offsets(path: Path) -> None:
    for index_file in experiment_index_files(path):
        index_data = read_json(index_file)
        experiment_data = index_data.get("experiment", {})

        for response_config in experiment_data.get("response_configuration", []):
            if response_config.get("type") == "everest_objectives":
                response_config["offsets"] = [0.0] * len(
                    response_config.get("weights", [])
                )

        write_json(index_file, index_data)


def migrate(path: Path) -> None:
    _add_objective_offsets(path)
