from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

info = "Move ES-MDA weights into experiment analysis_settings"


def _move_weights_into_analysis_settings(path: Path) -> None:
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

        if "weights" not in experiment_data:
            continue

        weights = experiment_data.pop("weights")
        analysis_settings = experiment_data.setdefault("analysis_settings", {})
        analysis_settings.setdefault("weights", weights)

        index_file.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
        logger.info("Moved weights into analysis_settings in %s", index_file)


def migrate(path: Path) -> None:
    _move_weights_into_analysis_settings(path)
