from pathlib import Path

import ert
import ert3


def status(workspace_root: Path) -> None:
    experiments = ert3.workspace.get_experiment_names(workspace_root)
    done = ert.storage.get_experiment_names(workspace=workspace_root)
    pending = [experiment for experiment in experiments if experiment not in done]

    if done:
        print("Experiments that have run already:")
        for experiment in done:
            print(f"  {experiment}")

    if pending:
        if done:
            print()
        print("Experiments that can be run:")
        for experiment in pending:
            print(f"  {experiment}")

    if not done and not pending:
        print("No experiments present in this workspace")
