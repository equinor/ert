import json
from pathlib import Path
from typing import Any

info = "Remove refcase from summary response configs"


def config_without_refcase(summary_config: dict[str, Any]) -> dict[str, Any]:
    new_json = {**summary_config}
    new_json.pop("refcase", None)

    return new_json


def migrate(path: Path) -> None:
    for response_json_path in path.glob("experiments/*/responses.json"):
        old_json = json.loads((response_json_path).read_text(encoding="utf-8"))
        new_json = {
            response_type: config_without_refcase(response_config)
            if response_config["type"] == "summary"
            else response_config
            for response_type, response_config in old_json.items()
        }

        response_json_path.write_text(json.dumps(new_json, indent=2), encoding="utf-8")
