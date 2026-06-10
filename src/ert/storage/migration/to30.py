from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

info = "Strip fields not in ExperimentConfig from experiment index"

_ALLOWED_EXPERIMENT_KEYS = frozenset(
    {
        "experiment_type",
        "ert_templates",
        "observations",
        "design_matrix",
        "parameter_configuration",
        "response_configuration",
        "derived_response_configuration",
        "target_ensemble",
        "shape_registry",
        "analysis_settings",
        "update_settings",
        "ensemble_id",
        "restart_run",
        "prior_ensemble_id",
        "weights",
        "optimization_output_dir",
        "simulation_dir",
        "input_constraints",
        "optimization",
        "model",
        "keep_run_path",
        "experiment_name",
    }
)


def _strip_unknown_fields(path: Path) -> None:
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

        unknown_keys = [
            key for key in experiment_data if key not in _ALLOWED_EXPERIMENT_KEYS
        ]
        if unknown_keys:
            for key in unknown_keys:
                experiment_data.pop(key)
            logger.info(
                "Stripped fields not in ExperimentConfig from %s: %s",
                index_file,
                ", ".join(sorted(unknown_keys)),
            )
            index_file.write_text(json.dumps(index_data, indent=2), encoding="utf-8")


def migrate(path: Path) -> None:
    _strip_unknown_fields(path)
