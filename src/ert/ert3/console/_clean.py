from typing import Set

import ert.storage
from ert import ert3


def clean(
    workspace: ert3.workspace.Workspace, experiment_names: Set[str], clean_all: bool
) -> None:
    if clean_all:
        non_existent = []
    else:
        non_existent = [
            name
            for name in experiment_names
            if name
            not in ert.storage.get_experiment_names(workspace_name=workspace.name)
        ]

    ert3.engine.clean(workspace, experiment_names, clean_all)

    if non_existent:
        print("Following experiment(s) did not exist:")
        for name in non_existent:
            print(f"    {name}")
        print("Perhaps you mistyped an experiment name?")
