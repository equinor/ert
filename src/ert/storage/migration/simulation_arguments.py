import json
from pathlib import Path

from ert.storage.local_experiment import LocalExperiment


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):

        if not experiment.joinpath(LocalExperiment._simulation_arguments_file).exists():
            with open(
                experiment.joinpath(LocalExperiment._simulation_arguments_file),
                mode="w",
                encoding="utf-8",
            ) as f:
                json.dump({}, f)
