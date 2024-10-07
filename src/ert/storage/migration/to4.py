from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ert.storage.local_storage import local_storage_get_ert_config

if TYPE_CHECKING:
    from pathlib import Path


info = "Introducing observations and removing template_file_path"


def migrate(path: Path) -> None:
    ert_config = local_storage_get_ert_config()
    observations = ert_config.observations

    for experiment in path.glob("experiments/*"):
        if observations:
            output_path = experiment / "observations"
            output_path.mkdir(parents=True, exist_ok=True)
            ert_config.enkf_obs.write_to_folder(dest=output_path)
        with open(experiment / "parameter.json", encoding="utf-8") as fin:
            parameters_json = json.load(fin)
        with open(experiment / "parameter.json", "w", encoding="utf-8") as fout:
            fout.write(json.dumps(parameters_json, indent=3))
