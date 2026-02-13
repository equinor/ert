from __future__ import annotations

import json
from pathlib import Path
from typing import Any

info = "Unroll EverestControl to one config per parameter"


def migrate_everest_control_format(path: Path) -> None:
    experiments_dir = path / "experiments"
    if not experiments_dir.exists():
        return

    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        index_file = exp_dir / "index.json"
        index_data = json.loads(index_file.read_text(encoding="utf-8"))

        experiment_data = index_data.get("experiment")
        params_config = experiment_data.get("parameter_configuration")
        new_params_config: list[dict[str, Any]] = []
        modified = False

        for param in params_config:
            if param.get("type") == "everest_parameters" and "input_keys" in param:
                modified = True

                common_fields = {
                    "forward_init": False,
                    "output_file": param["output_file"],
                    "forward_init_file": "",
                    "update": False,
                    "type": "everest_parameters",
                    "dimensionality": 1,
                }

                group_name = param["name"]
                input_keys = param["input_keys"]

                for i, input_key in enumerate(input_keys):
                    new_params_config.append(
                        {
                            **common_fields,
                            "input_key": input_key,
                            "group": group_name,
                            "name": input_key,
                            "control_type_": param["types"][i],
                            "initial_guess": param["initial_guesses"][i],
                            "variable_type": param["control_types"][i],
                            "enabled": param["enabled"][i],
                            "min": param["min"][i],
                            "max": param["max"][i],
                            "perturbation_type": param["perturbation_types"][i],
                            "perturbation_magnitude": param["perturbation_magnitudes"][
                                i
                            ],
                            "scaled_range": param["scaled_ranges"][i],
                            "sampler": param["samplers"][i],
                            "input_key_dotdash": param["input_keys_dotdash"][i],
                        }
                    )
            else:
                new_params_config.append(param)

        if modified:
            experiment_data["parameter_configuration"] = new_params_config
            index_file.write_text(json.dumps(index_data), encoding="utf-8")


def migrate(path: Path) -> None:
    migrate_everest_control_format(path)
