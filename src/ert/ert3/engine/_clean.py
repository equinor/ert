from typing import TYPE_CHECKING, Set

import ert.storage

if TYPE_CHECKING:
    from ert.ert3.workspace import Workspace


def clean(workspace: "Workspace", experiment_names: Set[str], clean_all: bool) -> None:
    assert not (experiment_names and clean_all)

    stored_experiments = ert.storage.get_experiment_names(workspace_name=workspace.name)

    if clean_all:
        experiment_names = stored_experiments
    else:
        experiment_names = {
            name for name in experiment_names if name in stored_experiments
        }

    for name in experiment_names:
        ert.storage.delete_experiment(experiment_name=name)
