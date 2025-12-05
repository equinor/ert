import json
from pathlib import Path
from typing import Any

info = "Add dimensionality attribute to parameters"


def migrate_param(parameters_json: dict[str, Any]) -> dict[str, Any]:
    new_configs = {}
    for param_config in parameters_json.values():
        if param_config["type"] == "surface":
            param_config["dimensionality"] = 2
        elif param_config["type"] == "field":
            param_config["dimensionality"] = 3
        else:
            param_config["dimensionality"] = 1

        new_configs[param_config["name"]] = param_config
    return new_configs


def migrate_parameters_for_experiment(experiment: Path) -> None:
    with open(experiment / "parameter.json", encoding="utf-8") as fin:
        parameters_json = json.load(fin)

    new_parameter_configs = migrate_param(parameters_json)
    Path(experiment / "parameter.json").write_text(
        json.dumps(new_parameter_configs, indent=2), encoding="utf-8"
    )


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        migrate_parameters_for_experiment(experiment)
