from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

info = "Unify response_configuration field with derived_response_configuration field"


def _move_derived_response_configuration_to_response_configuration(path: Path) -> None:
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

        response_config = experiment_data.get("response_configuration", [])
        derived_response_config = experiment_data.pop(
            "derived_response_configuration", []
        )

        if derived_response_config:
            experiment_data["response_configuration"] = (
                response_config + derived_response_config
            )

        index_file.write_text(json.dumps(index_data, indent=2), encoding="utf-8")


def migrate(path: Path) -> None:
    _move_derived_response_configuration_to_response_configuration(path)
