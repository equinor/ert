import ert3.console
import ert3.storage
import ert3.evaluator
import ert3.engine

from pathlib import Path


_WORKSPACE_DATA_ROOT = ".ert"


def _locate_ert_workspace_root(path):
    path = Path(path)
    while True:
        if (path / ert3._WORKSPACE_DATA_ROOT).exists():
            return path
        if path == Path(path.root):
            return None
        path = path.parent


def _assert_experiment(workspace_root, experiment_name):
    experiment_root = Path(workspace_root) / experiment_name
    if not experiment_root.is_dir():
        raise ValueError(
            f"{experiment_name} is not an experiment "
            f"within the workspace {workspace_root}"
        )


def _experiment_have_run(workspace_root, experiment_name):
    experiments = ert3.storage.get_experiment_names(workspace_root)
    return experiment_name in experiments
