"""
_workspace.py
====================================
The workspace holds all knowledge of config(s) and filesystem.
It's knowledge does not leak into other parts of the code.
It may only be employed.
Is elsewise stateless, and is independent of storage.
"""


import json
import sys
import shutil
from pathlib import Path
from typing import Union, Optional, Set, List, Dict, Any, Tuple

import yaml

import ert3
import ert

_WORKSPACE_DATA_ROOT = ".ert"
_EXPERIMENTS_BASE = "experiments"


def _locate_root(path: Path) -> Optional[Path]:
    while True:
        if (path / _WORKSPACE_DATA_ROOT).exists():
            return path
        if path == Path(path.root):
            return None
        path = path.parent


class Workspace:
    """A Workspace holds the knowledge of experiments, resources and config"""

    def __init__(self, path: Union[str, Path]) -> None:
        root = _locate_root(Path(path))
        if root is None:
            raise ert.exceptions.IllegalWorkspaceOperation(
                "Not inside an ERT workspace."
            )
        self._path = root

    @property
    def name(self) -> str:
        return str(self._path)

    def get_experiment_tmp_dir(self, experiment_name: str) -> Path:
        if experiment_name not in self.get_experiment_names():
            raise ert.exceptions.IllegalWorkspaceOperation(
                f"experiment {experiment_name} does not exist"
            )
        return self._path / _WORKSPACE_DATA_ROOT / "tmp" / experiment_name

    def get_resources_dir(self) -> Path:
        return self._path / "resources"

    def assert_experiment_exists(self, experiment_name: str) -> None:
        """Check if the experiment `experiment_name` exists in workspace."""

        experiment_root = self._path / _EXPERIMENTS_BASE / experiment_name
        if not experiment_root.is_dir():
            raise ert.exceptions.IllegalWorkspaceOperation(
                f"{experiment_name} is not an experiment "
                f"within the workspace {self.name}"
            )

    def get_experiment_names(self) -> Set[str]:
        experiment_base = self._path / _EXPERIMENTS_BASE
        if not experiment_base.is_dir():
            raise ert.exceptions.IllegalWorkspaceState(
                f"the workspace {self._path} cannot access experiments"
            )
        return {
            experiment.name
            for experiment in experiment_base.iterdir()
            if experiment.is_dir()
        }

    def load_experiment_config(
        self, experiment_name: str
    ) -> Tuple[
        ert3.config.ExperimentConfig,
        ert3.config.StagesConfig,
        ert3.config.EnsembleConfig,
    ]:
        experiment_config_path = (
            self._path / _EXPERIMENTS_BASE / experiment_name / "experiment.yml"
        )
        with open(experiment_config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        experiment_config = ert3.config.load_experiment_config(config_dict)

        stages_config_path = self._path / "stages.yml"
        with open(stages_config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        sys.path.append(str(self._path))
        stage_config = ert3.config.load_stages_config(config_dict)

        ensemble_config_path = (
            self._path / _EXPERIMENTS_BASE / experiment_name / "ensemble.yml"
        )
        with open(ensemble_config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        ensemble_config = ert3.config.load_ensemble_config(config_dict)

        _validate_ensemble_size(experiment_config, ensemble_config)
        _validate_stage(stage_config, ensemble_config)

        return experiment_config, stage_config, ensemble_config

    def load_parameters_config(self) -> ert3.config.ParametersConfig:
        with open(self._path / "parameters.yml", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return ert3.config.load_parameters_config(config_dict)

    def clean_experiment(self, experiment_name: str) -> None:
        tmp_dir = self.get_experiment_tmp_dir(experiment_name)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    def export_json(
        self,
        experiment_name: str,
        data: Union[Dict[int, Dict[str, Any]], List[Dict[str, Dict[str, Any]]]],
        output_file: Optional[str] = None,
    ) -> None:
        experiment_root = self._path / _EXPERIMENTS_BASE / experiment_name
        if output_file is None:
            output_file = "data.json"
        self.assert_experiment_exists(experiment_name)
        with open(experiment_root / output_file, "w", encoding="utf-8") as f:
            json.dump(data, f)


def initialize(path: Union[str, Path]) -> Workspace:
    path = Path(path)
    if _locate_root(path) is not None:
        raise ert.exceptions.IllegalWorkspaceOperation(
            "Already inside an ERT workspace."
        )
    (path / _WORKSPACE_DATA_ROOT).mkdir()
    return Workspace(path)


def _validate_ensemble_size(
    experiment_config: ert3.config.ExperimentConfig,
    ensemble_config: ert3.config.EnsembleConfig,
) -> None:
    if experiment_config.type == "sensitivity" and ensemble_config.size is not None:
        raise ert.exceptions.ConfigValidationError(
            "No ensemble size should be specified for a sensitivity analysis."
        )
    if experiment_config.type != "sensitivity" and ensemble_config.size is None:
        raise ert.exceptions.ConfigValidationError(
            "An ensemble size must be specified."
        )


def _validate_stage(
    stage_config: ert3.config.StagesConfig, ensemble_config: ert3.config.EnsembleConfig
) -> None:
    stage_name = ensemble_config.forward_model.stage
    stage = next((stage for stage in stage_config if stage.name == stage_name), None)
    if stage is None:
        raise ert.exceptions.ConfigValidationError(
            f"Invalid stage in forward model: '{stage_name}'. "
            f"Must be one of: " + ", ".join(f"'{stage.name}'" for stage in stage_config)
        )
    stage_input_names = set(stage.input.keys())
    ensemble_input_names = set(input.record for input in ensemble_config.input)
    if ensemble_input_names != stage_input_names:
        raise ert.exceptions.ConfigValidationError(
            "Ensemble and stage inputs do not match."
        )
