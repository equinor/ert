from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


from ert.storage.local_storage import local_storage_get_ert_config

info = "Introducing ert_kind, adding responses.json and removing template_file_path"


def migrate(path: Path) -> None:
    ert_config = local_storage_get_ert_config()
    ens_config = ert_config.ensemble_config
    for experiment in path.glob("experiments/*"):
        with open(experiment / "parameter.json", encoding="utf-8") as fin:
            parameters_json = json.load(fin)
        key_to_remove = "template_file_path"
        for config in parameters_json.values():
            if key_to_remove in config:
                del config[key_to_remove]
            if "mask_file" in config:
                config["_ert_kind"] = "Field"
            elif "base_surface_path" in config:
                config["_ert_kind"] = "SurfaceConfig"
            elif "template_file" in config:
                config["_ert_kind"] = "GenKwConfig"
        with open(experiment / "parameter.json", "w", encoding="utf-8") as fout:
            fout.write(json.dumps(parameters_json, indent=4))

        response_info = {}
        for response in ens_config.response_configuration:
            response_info[response.name] = response.model_dump() | {
                "_ert_kind": response.__class__.__name__
            }
        with open(experiment / "responses.json", "w", encoding="utf-8") as fout:
            fout.write(json.dumps(response_info, default=str, indent=4))
