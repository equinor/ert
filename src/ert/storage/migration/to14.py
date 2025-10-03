import json
from pathlib import Path
from typing import Any

info = "Add ertbox_params to field config"


def migrate_field_param(parameters_json: dict[str, Any]) -> dict[str, Any]:
    new_configs = {}
    for param_config in parameters_json.values():
        if param_config["type"] == "field":
            ertbox_params = {}
            ertbox_params["nx"] = param_config["nx"]
            ertbox_params["ny"] = param_config["ny"]
            ertbox_params["nz"] = param_config["nz"]
            ertbox_params["xlength"] = None
            ertbox_params["ylength"] = None
            ertbox_params["xinc"] = None
            ertbox_params["yinc"] = None
            ertbox_params["rotation_angle"] = None
            ertbox_params["origin"] = None
            param_config["ertbox_params"] = ertbox_params
            del param_config["nx"]
            del param_config["ny"]
            del param_config["nz"]

        new_configs[param_config["name"]] = param_config
    return new_configs


def migrate_fields(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        with open(experiment / "parameter.json", encoding="utf-8") as fin:
            parameters_json = json.load(fin)

        new_parameter_configs = migrate_field_param(parameters_json)
        with open(experiment / "parameter.json", "w", encoding="utf-8") as fout:
            fout.write(json.dumps(new_parameter_configs, indent=2))


def migrate(path: Path) -> None:
    migrate_fields(path)
