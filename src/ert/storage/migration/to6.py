import json
from pathlib import Path

info = "Rename and change transfer_function_definitions"


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        parameters_json = json.loads(
            (experiment / "parameter.json").read_text(encoding="utf-8")
        )

        for param in parameters_json.values():
            if "transfer_function_definitions" in param:
                param["transform_function_definitions"] = param[
                    "transfer_function_definitions"
                ]
                del param["transfer_function_definitions"]

            if "transform_function_definitions" in param:
                transform_function_definitions = []
                for tfd in param["transform_function_definitions"]:
                    if isinstance(tfd, str):
                        items = tfd.split()
                        transform_function_definitions.append(
                            {
                                "name": items[0],
                                "param_name": items[1],
                                "values": items[2:],
                            }
                        )
                    elif isinstance(tfd, dict):
                        transform_function_definitions.append(tfd)

                param["transform_function_definitions"] = transform_function_definitions
        Path(experiment / "parameter.json").write_text(
            json.dumps(parameters_json, indent=4), encoding="utf-8"
        )
