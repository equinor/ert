import ert3

from pathlib import Path
import sys


def init(path):
    path = Path(path)
    if locate_root(path) is not None:
        sys.exit("Already inside an ERT workspace")

    ert3.storage.init(path)


def locate_root(path):
    path = Path(path)
    while True:
        if (path / ert3._WORKSPACE_DATA_ROOT).exists():
            return path
        if path == Path(path.root):
            return None
        path = path.parent


def experiment_exists(workspace_root, experiment_name):
    experiment_root = Path(workspace_root) / experiment_name
    if not experiment_root.is_dir():
        raise ValueError(
            f"{experiment_name} is not an experiment "
            f"within the workspace {workspace_root}"
        )


def experiment_have_run(workspace_root, experiment_name):
    experiments = ert3.storage.get_experiment_names(workspace_root)
    return experiment_name in experiments
