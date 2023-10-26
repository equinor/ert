from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ert.storage.local_storage import local_storage_get_ert_config

if TYPE_CHECKING:
    from pathlib import Path


def migrate(path: Path) -> None:
    ert_config = local_storage_get_ert_config()
    observations = ert_config.observations

    for experiment in path.glob("experiments/*"):
        if observations:
            output_path = experiment / "observations"
            output_path.mkdir(parents=True, exist_ok=True)
            for name, dataset in observations.items():
                dataset.to_netcdf(output_path / f"{name}", engine="scipy")

        with open(path / "index.json", encoding="utf-8") as f:
            index_json = json.load(f)
        index_json["version"] = 4
        with open(path / "index.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(index_json))
