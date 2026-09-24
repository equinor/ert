from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

info = "Add offsets to the Everest objectives response configuration"


def _add_objective_offsets(path: Path) -> None:
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

        for response_config in experiment_data.get("response_configuration", []):
            if response_config.get("type") == "everest_objectives":
                response_config["offsets"] = [0.0] * len(
                    response_config.get("weights", [])
                )

        index_file.write_text(json.dumps(index_data, indent=2), encoding="utf-8")


def migrate(path: Path) -> None:
    _add_objective_offsets(path)
