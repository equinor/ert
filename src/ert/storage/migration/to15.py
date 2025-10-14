import json
from pathlib import Path
from typing import Any

info = "Add ertbox_params to field config"


def migrate_field_param(parameters_json: dict[str, Any]) -> dict[str, Any]:
    new_configs = {}
    for param_config in parameters_json.values():
        if param_config["type"] == "field":
            del param_config["mask_file"]

        new_configs[param_config["name"]] = param_config
    return new_configs


def migrate_fields(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        with open(experiment / "parameter.json", encoding="utf-8") as fin:
            parameters_json = json.load(fin)

        new_parameter_configs = migrate_field_param(parameters_json)
        Path(experiment / "parameter.json").write_text(
            json.dumps(new_parameter_configs, indent=2), encoding="utf-8"
        )

        # Delete grid_mask.npy if it exists
        grid_mask_file = experiment / "grid_mask.npy"
        if grid_mask_file.exists():
            grid_mask_file.unlink()


def migrate(path: Path) -> None:
    migrate_fields(path)
