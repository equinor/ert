from __future__ import annotations

import logging
from pathlib import Path

from .migration_utils import experiment_index_files, read_json, write_json

logger = logging.getLogger(__name__)

info = "Unify response_configuration field with derived_response_configuration field"


def _move_derived_response_configuration_to_response_configuration(path: Path) -> None:
    for index_file in experiment_index_files(path):
        index_data = read_json(index_file)
        experiment_data = index_data.get("experiment", {})

        response_config = experiment_data.get("response_configuration", [])
        derived_response_config = experiment_data.pop(
            "derived_response_configuration", []
        )

        if derived_response_config:
            experiment_data["response_configuration"] = (
                response_config + derived_response_config
            )

        write_json(index_file, index_data)


def migrate(path: Path) -> None:
    _move_derived_response_configuration_to_response_configuration(path)
