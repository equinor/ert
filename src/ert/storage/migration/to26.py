from __future__ import annotations

from pathlib import Path
from typing import Any

from .migration_utils import experiment_dirs, read_json, write_json

info = "Unroll EverestControl to one config per parameter"


def migrate_everest_control_format(path: Path) -> None:
    for exp_dir in experiment_dirs(path):
        index_file = exp_dir / "index.json"
        index_data = read_json(index_file)

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
                            "control_type": param["control_types"][i],
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
            write_json(index_file, index_data)


def migrate(path: Path) -> None:
    migrate_everest_control_format(path)
