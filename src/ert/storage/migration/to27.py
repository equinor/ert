import json
from pathlib import Path
from typing import Any

info = "Rename ertbox to grid dimensions"


def migrate_field_param(param_config: dict[str, Any]) -> dict[str, Any]:
    new_config = param_config.copy()
    if param_config["type"] == "field":
        new_config["grid_geometry"] = param_config.pop("ertbox_params", None)
    return new_config


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        index_file = experiment / "index.json"
        index_data = json.loads(index_file.read_text(encoding="utf-8"))

        experiment_data = index_data.get("experiment")
        parameters_config = experiment_data.get("parameter_configuration")
        new_configs = []
        for parameter in parameters_config:
            new_configs.append(migrate_field_param(parameter))
        experiment_data["parameter_configuration"] = new_configs
        index_file.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
