from pathlib import Path
from uuid import UUID

from ert.storage.local_experiment import _Index


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        if not experiment.joinpath("index.json").exists():
            with open(
                experiment.joinpath("index.json"), mode="w", encoding="utf-8"
            ) as f:
                print(
                    _Index(
                        id=UUID(experiment.name), name="default_experiment_name"
                    ).model_dump_json(),
                    file=f,
                )
