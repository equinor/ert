import json
from pathlib import Path
from typing import Any

info = "Remove redundant .name attribute from responses."


def config_without_name_attr(config: dict[str, Any]) -> dict[str, Any]:
    new_json = {**config}
    new_json.pop("name", None)

    return new_json


def migrate(path: Path) -> None:
    for response_json_path in path.glob("experiments/*/responses.json"):
        old_json = json.loads((response_json_path).read_text(encoding="utf-8"))
        new_json = {
            response_type: config_without_name_attr(config)
            for response_type, config in old_json.items()
        }

        response_json_path.write_text(json.dumps(new_json, indent=2), encoding="utf-8")
