import json
import pathlib
from pathlib import Path
from typing import Any

info = (
    "Add status field to experiment index.json files. "
    "Move metadata.json contents to be in experiment index.json .experiment field"
)


def add_experiment_status_to_index_json(index_json: dict[str, Any]) -> dict[str, Any]:
    index_json["status"] = {"message": "", "status": "completed"}
    return index_json


def migrate_index_json(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        index_path = experiment / "index.json"
        index_data = json.loads(pathlib.Path(index_path).read_text(encoding="utf-8"))
        pathlib.Path(index_path).write_text(
            json.dumps(add_experiment_status_to_index_json(index_data), indent=2),
            encoding="utf-8",
        )


def migrate_metadata_json_into_experiment_index(path: Path) -> None:
    for experiment_path in path.glob("experiments/*"):
        experiment_json = {}
        with open(experiment_path / "metadata.json", encoding="utf-8") as fin:
            metadata_json = json.load(fin)
            if "weights" in metadata_json:
                experiment_json["weights"] = metadata_json["weights"]

        with open(experiment_path / "index.json", encoding="utf-8") as fin:
            index_json = json.load(fin)
            index_json["experiment"] = experiment_json

            Path(experiment_path / "index.json").write_text(
                json.dumps(index_json, indent=2), encoding="utf-8"
            )

        (experiment_path / "metadata.json").unlink()


def migrate(path: Path) -> None:
    migrate_index_json(path)
    migrate_metadata_json_into_experiment_index(path)
