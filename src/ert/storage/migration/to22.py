import json
from pathlib import Path
from typing import Any

info = "Rename ertbox to grid dimensions"


def migrate_field_param(parameters_json: dict[str, Any]) -> dict[str, Any]:
    new_configs = {}
    for param_config in parameters_json.values():
        if param_config["type"] == "field":
            param_config["grid_geometry"] = param_config.pop("ertbox_params")

        new_configs[param_config["name"]] = param_config
    return new_configs


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        parameters_json = json.loads(
            (experiment / "parameter.json").read_text(encoding="utf-8")
        )
        new_parameter_configs = migrate_field_param(parameters_json)
        Path(experiment / "parameter.json").write_text(
            json.dumps(new_parameter_configs, indent=2), encoding="utf-8"
        )
