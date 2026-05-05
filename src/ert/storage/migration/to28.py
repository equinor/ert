from __future__ import annotations

import json
from pathlib import Path

info = "Migrate 'update' field from bool to strategy string"


def _migrate_update_bool_to_strategy(path: Path) -> None:
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
        params_config = experiment_data.get("parameter_configuration", [])

        modified = False
        for param in params_config:
            if "update" in param and isinstance(param["update"], bool):
                param["update"] = "ADAPTIVE" if param["update"] else None
                modified = True

        if modified:
            index_file.write_text(json.dumps(index_data, indent=2), encoding="utf-8")


def migrate(path: Path) -> None:
    _migrate_update_bool_to_strategy(path)
