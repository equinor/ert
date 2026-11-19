from __future__ import annotations

from pathlib import Path

from ert.config.parameter_config import LocalizationType

from .migration_utils import experiment_index_files, read_json, write_json

info = "Migrate 'update' field from bool to strategy string"


def _migrate_update_bool_to_strategy(path: Path) -> None:
    for index_file in experiment_index_files(path):
        index_data = read_json(index_file)
        experiment_data = index_data.get("experiment", {})
        params_config = experiment_data.get("parameter_configuration", [])

        modified = False
        for param in params_config:
            if "update" in param and isinstance(param["update"], bool):
                param["update_strategy"] = (
                    LocalizationType.GLOBAL.value if param["update"] else None
                )
                modified = True
                param.pop("update")

        if modified:
            write_json(index_file, index_data)


def migrate(path: Path) -> None:
    _migrate_update_bool_to_strategy(path)
