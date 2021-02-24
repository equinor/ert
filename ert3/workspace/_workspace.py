import ert3

from pathlib import Path


def _locate_root(path):
    path = Path(path)
    while True:
        if (path / ert3._WORKSPACE_DATA_ROOT).exists():
            return path
        if path == Path(path.root):
            return None
        path = path.parent


def assert_experiment_exists(workspace_root, experiment_name):
    experiment_root = (
        Path(workspace_root) / ert3.workspace.EXPERIMENTS_BASE / experiment_name
    )
    if not experiment_root.is_dir():
        raise ert3.exceptions.IllegalWorkspaceOperation(
            f"{experiment_name} is not an experiment "
            f"within the workspace {workspace_root}"
        )


def experiment_have_run(workspace_root, experiment_name):
    experiments = ert3.storage.get_experiment_names(workspace=workspace_root)
    return experiment_name in experiments


def initialize(path):
    path = Path(path)
    if load(path) is not None:
        raise ert3.exceptions.IllegalWorkspaceOperation(
            "Already inside an ERT workspace."
        )

    ert3.storage.init(workspace=path)


def load(path):
    return _locate_root(path)
