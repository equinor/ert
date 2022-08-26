from typing import Set

import ert.storage
from ert.ert3 import engine
from ert.ert3.workspace import Workspace


def clean(workspace: Workspace, experiment_names: Set[str], clean_all: bool) -> None:
    if clean_all:
        non_existent = []
    else:
        non_existent = [
            name
            for name in experiment_names
            if name
            not in ert.storage.get_experiment_names(workspace_name=workspace.name)
        ]

    engine.clean(workspace, experiment_names, clean_all)

    if non_existent:
        print("Following experiment(s) did not exist:")
        for name in non_existent:
            print(f"    {name}")
        print("Perhaps you mistyped an experiment name?")
