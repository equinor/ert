from ert3.workspace._workspace import initialize
from ert3.workspace._workspace import load
from ert3.workspace._workspace import get_experiment_names
from ert3.workspace._workspace import experiment_has_run
from ert3.workspace._workspace import assert_experiment_exists

EXPERIMENTS_BASE = "experiments"

__all__ = [
    "initialize",
    "load",
    "get_experiment_names",
    "experiment_has_run",
    "assert_experiment_exists",
]
