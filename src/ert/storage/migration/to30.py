from __future__ import annotations

import logging
from pathlib import Path

from .migration_utils import experiment_index_files, read_json, write_json

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
        "minimum_required_realizations",
        "random_seed",
        "storage_path",
        "user_config_file",
        "runpath_file",
        "log_path",
        "start_iteration",
        "active_realizations",
        "hooked_workflows",
        "substitutions",
        "queue_config",
        "env_vars",
        "env_pr_fm_step",
        "runpath_config",
        "forward_model_steps",
    }
)


def _strip_unknown_fields(path: Path) -> None:
    for index_file in experiment_index_files(path):
        index_data = read_json(index_file)
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
            write_json(index_file, index_data)


def migrate(path: Path) -> None:
    _strip_unknown_fields(path)
