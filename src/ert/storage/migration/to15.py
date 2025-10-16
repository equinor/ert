import json
from pathlib import Path
from typing import Any

info = "Add status field to experiment index.json files"


def update_json(index_json: dict[str, Any]) -> dict[str, Any]:
    index_json["status"] = {"message": "", "status": "completed"}
    return index_json


def migrate_index_json(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        index_path = experiment / "index.json"
        with open(index_path, encoding="utf-8") as f:
            index_data = json.load(f)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(update_json(index_data), f, indent=2)


def migrate(path: Path) -> None:
    migrate_index_json(path)
