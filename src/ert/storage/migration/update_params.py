from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        with open(experiment / "parameter.json", encoding="utf-8") as fin:
            parameters_json = json.load(fin)
        for param_group in parameters_json:
            if "update" not in parameters_json[param_group]:
                parameters_json[param_group]["update"] = True
        with open(experiment / "parameter.json", "w", encoding="utf-8") as fout:
            fout.write(json.dumps(parameters_json))
