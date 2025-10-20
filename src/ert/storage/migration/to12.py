import json
from pathlib import Path
from typing import Any

info = "Migrate response and parameter configs to include type"


def migrate_everest_param(config: dict[str, Any]) -> dict[str, Any]:
    formatted_control_names = []

    name = config["name"]
    input_keys = config["input_keys"]

    if isinstance(input_keys, list):
        return config

    # It is a dict
    assert isinstance(input_keys, dict)
    for k, v in input_keys.items():
        for subkey in v:
            formatted_control_names.append(f"{name}.{k}.{subkey}")

    return config | {"input_keys": formatted_control_names}


def migrate(path: Path) -> None:
    def _replace_ert_kind(file: Path, kind_to_type: dict[str, str]) -> None:
        old_json = json.loads(file.read_text(encoding="utf-8"))
        new_json = {}

        for key, config in old_json.items():
            ert_kind = config.pop("_ert_kind")

            if ert_kind == "ExtParamConfig":
                new_json[key] = migrate_everest_param(config) | {
                    "type": "everest_parameters"
                }
            else:
                new_json[key] = config | {"type": kind_to_type[ert_kind]}

        file.write_text(json.dumps(new_json, indent=2), encoding="utf-8")

    for experiment in path.glob("experiments/*"):
        _replace_ert_kind(
            file=experiment / "parameter.json",
            kind_to_type={
                "GenKwConfig": "gen_kw",
                "ExtParamConfig": "everest_parameters",
                "Field": "field",
                "SurfaceConfig": "surface",
            },
        )

        _replace_ert_kind(
            file=experiment / "responses.json",
            kind_to_type={
                "GenDataConfig": "gen_data",
                "SummaryConfig": "summary",
                "EverestConstraintsConfig": "everest_constraints",
                "EverestObjectivesConfig": "everest_objectives",
            },
        )

        old_json = json.loads(
            (experiment / "responses.json").read_text(encoding="utf-8")
        )
        new_json = {}

        for key, config in old_json.items():
            if config["type"] == "summary" and "refcase" in config:
                config.pop("refcase")

            new_json[key] = config
