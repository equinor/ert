from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from ert.config import (
    SCALAR_PARAMETERS_NAME,
    DataSource,
    ScalarParameter,
    ScalarParameters,
    get_distribution,
)

info = "Introducing ScalarParameters replacing GenKWConfig"


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        with open(experiment / "parameter.json", encoding="utf-8") as fin:
            parameters_json = json.load(fin)
        gen_kw_params = {
            config["name"]: config
            for config in parameters_json.values()
            if config["_ert_kind"] == "GenKwConfig"
        }
        scalars: list[ScalarParameter] = []
        for group, config in gen_kw_params.items():
            for param in config["transform_function_definitions"]:
                scalars.append(
                    ScalarParameter(
                        param_name=param["name"],
                        group_name=group,
                        distribution=get_distribution(
                            param["param_name"], param["values"]
                        ),
                        input_source=DataSource.SAMPLED,
                        update=config["update"],
                        output_file=config["output_file"],
                        template_file=config["template_file"],
                    )
                )
            del parameters_json[group]
        sc = ScalarParameters(scalars=scalars)
        sc.save_experiment_data(experiment)
        parameters_json[SCALAR_PARAMETERS_NAME] = sc.to_dict()
        with open(experiment / "parameter.json", "w", encoding="utf-8") as fout:
            fout.write(json.dumps(parameters_json, indent=4))
