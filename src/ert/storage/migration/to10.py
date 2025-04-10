import json
from pathlib import Path

info = "Migrate template GEN_KW feature into RUN_TEMPLATE"


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        with open(experiment / "parameter.json", encoding="utf-8") as fin:
            parameters_json = json.load(fin)

        with open(experiment / "parameter.json", "w", encoding="utf-8") as fout:
            for param in parameters_json.values():
                if param["_ert_kind"] == "GenKwConfig":
                    param.pop("template_file", None)
                    param.pop("output_file", None)
                    param.pop("forward_init_file", None)
            fout.write(json.dumps(parameters_json, indent=4))
