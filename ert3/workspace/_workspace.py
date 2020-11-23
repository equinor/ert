import ert3

from pathlib import Path
import sys


def _locate_root(path):
    path = Path(path)
    while True:
        if (path / ert3._WORKSPACE_DATA_ROOT).exists():
            return path
        if path == Path(path.root):
            return None
        path = path.parent


class _Workspace:
    def __init__(self, path):
        self._root = _locate_root(path)
        if root is None:
            raise ValueError(f"{path} is not inside an ERT workspace")

    @property
    def root(self):
        return self._root

    def load_experiments(self):
        experiment_names = next(os.walk(self._root))[1]
        return [_Experiment(name, self.root, self.root / name) for name in experiment_names]

    def load_experiment(self, name):
        experiments = self.load_experiments()
        for experiment in experiments:
            if experiment.name == name:
                return experiment
        raise KeyError(f"No experiment named {name} in workspace {self.root}")

    def is_experiment(self, experiment_name):
        return experiment_name in [e.name for e in self.load_experiments()]

    def init_experiment(self, experiment_name):
        if self.is_experiment(experiment_name):
            raise ValueError(
                f"Cannot init experiment {experiment_name}, it already exists!"
            )

        ert3.storage.init_experiment(experiment_name)
        return self.load_experiment(experiment_name)


class _Experiment:
    def __init__(self, name, workspace_root, location):
        self._name = name
        self._workspace_root = workspace_root
        self._location = location

    @property
    def name(self):
        return self._name

    @property
    def workspace_root(self):
        return self._workspace_root

    @property
    def location(self):
        return self._location


def initialize(path):
    path = Path(path)
    if _locate_root(path) is not None:
        raise ValueError("Already inside an ERT workspace")

    ert3.storage.init(path)
    return _Workspace(path)


def load(path):
    return _Workspace(path)
