import json
from pathlib import Path

info = "Add design field into GenKwConfig"


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        with open(experiment / "parameter.json", encoding="utf-8") as fin:
            parameters_json = json.load(fin)

        with open(experiment / "parameter.json", "w", encoding="utf-8") as fout:
            for param in parameters_json.values():
                if param.get("forward_init") == True:
                    param["source"] = "forward_init"
                else:
                    param["source"] = "sampled"
                del param["forward_init"]
            fout.write(json.dumps(parameters_json, indent=4))
