from pathlib import Path

from ert3.exceptions import IllegalWorkspaceOperation, IllegalWorkspaceState
import ert3.storage as ert3_storage

EXPERIMENTS_BASE = "experiments"
DATA_ROOT = ".ert"


def _locate_root(path):
    path = Path(path)
    while True:
        if (path / DATA_ROOT).exists():
            return path
        if path == Path(path.root):
            return None
        path = path.parent


def assert_experiment_exists(workspace_root, experiment_name):
    experiment_root = (
        Path(workspace_root) / EXPERIMENTS_BASE / experiment_name
    )
    if not experiment_root.is_dir():
        raise IllegalWorkspaceOperation(
            f"{experiment_name} is not an experiment "
            f"within the workspace {workspace_root}"
        )


def get_experiment_names(workspace_root):
    experiment_base = Path(workspace_root) / EXPERIMENTS_BASE
    if not experiment_base.is_dir():
        raise IllegalWorkspaceState(
            f"the workspace {workspace_root} cannot access experiments"
        )
    return {
        experiment.name
        for experiment in experiment_base.iterdir()
        if experiment.is_dir()
    }


def experiment_has_run(workspace_root, experiment_name):
    experiments = ert3_storage.get_experiment_names(workspace=workspace_root)
    return experiment_name in experiments


def initialize(path):
    path = Path(path)
    if load(path) is not None:
        raise IllegalWorkspaceOperation(
            "Already inside an ERT workspace."
        )

    ert3_storage.init(workspace=path)


def load(path):
    return _locate_root(path)
