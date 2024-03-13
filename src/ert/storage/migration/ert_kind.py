from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        with open(experiment / "parameter.json", encoding="utf-8") as fin:
            parameters_json = json.load(fin)
        for config in parameters_json.values():
            if "mask_file" in config:
                config["_ert_kind"] = "Field"
            elif "base_surface_path" in config:
                config["_ert_kind"] = "SurfaceConfig"
            elif "template_file" in config:
                config["_ert_kind"] = "GenKwConfig"
        with open(experiment / "parameter.json", "w", encoding="utf-8") as fout:
            fout.write(json.dumps(parameters_json, sort_keys=True))

    with open(path / "index.json", encoding="utf-8") as f:
        index_json = json.load(f)
    index_json["version"] = 2
    with open(path / "index.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(index_json))
