import ert.storage
from ert import ert3


def status(workspace: ert3.workspace.Workspace) -> None:
    experiments = workspace.get_experiment_names()
    done = ert.storage.get_experiment_names(workspace_name=workspace.name)
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
