import json
import pathlib
from pathlib import Path
from typing import Any

info = "Add status field to experiment index.json files"


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


def migrate(path: Path) -> None:
    migrate_index_json(path)
