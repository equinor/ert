import os
import json
import sys
from pathlib import Path
from typing import Union, Optional, Set, List, Dict, Any

import yaml

import ert3
import ert

_EXPERIMENTS_BASE = "experiments"


def _locate_root(path: Union[str, Path]) -> Optional[Path]:
    path = Path(path)
    while True:
        if (path / ert3._WORKSPACE_DATA_ROOT).exists():
            return path
        if path == Path(path.root):
            return None
        path = path.parent


def assert_experiment_exists(
    workspace_root: Union[str, Path], experiment_name: str
) -> None:
    experiment_root = Path(workspace_root) / _EXPERIMENTS_BASE / experiment_name
    if not experiment_root.is_dir():
        raise ert.exceptions.IllegalWorkspaceOperation(
            f"{experiment_name} is not an experiment "
            f"within the workspace {workspace_root}"
        )


def get_experiment_names(workspace_root: Union[str, Path]) -> Set[str]:
    experiment_base = Path(workspace_root) / _EXPERIMENTS_BASE
    if not experiment_base.is_dir():
        raise ert.exceptions.IllegalWorkspaceState(
            f"the workspace {workspace_root} cannot access experiments"
        )
    return {
        experiment.name
        for experiment in experiment_base.iterdir()
        if experiment.is_dir()
    }


def initialize(path: Union[str, Path]) -> None:
    path = Path(path)
    if load(path) is not None:
        raise ert.exceptions.IllegalWorkspaceOperation(
            "Already inside an ERT workspace."
        )

    os.mkdir(path / ert3._WORKSPACE_DATA_ROOT)


def load(path: Union[str, Path]) -> Optional[Path]:
    return _locate_root(path)


def export_json(
    workspace_root: Path,
    experiment_name: str,
    data: Union[Dict[int, Dict[str, Any]], List[Dict[str, Dict[str, Any]]]],
    output_file: Optional[str] = None,
) -> None:
    experiment_root = Path(workspace_root) / _EXPERIMENTS_BASE / experiment_name
    if output_file is None:
        output_file = "data.json"
    assert_experiment_exists(workspace_root, experiment_name)
    with open(experiment_root / output_file, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_ensemble_config(
    workspace: Path, experiment_name: str
) -> ert3.config.EnsembleConfig:
    ensemble_config = workspace / _EXPERIMENTS_BASE / experiment_name / "ensemble.yml"
    with open(ensemble_config, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return ert3.config.load_ensemble_config(config_dict)


def load_stages_config(workspace: Path) -> ert3.config.StagesConfig:
    with open(workspace / "stages.yml", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    sys.path.append(str(workspace))
    return ert3.config.load_stages_config(config_dict)


def load_experiment_config(
    workspace: Path, experiment_name: str
) -> ert3.config.ExperimentConfig:
    experiment_config = (
        workspace / _EXPERIMENTS_BASE / experiment_name / "experiment.yml"
    )
    with open(experiment_config, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return ert3.config.load_experiment_config(config_dict)


def load_parameters_config(workspace: Path) -> ert3.config.ParametersConfig:
    with open(workspace / "parameters.yml", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return ert3.config.load_parameters_config(config_dict)
