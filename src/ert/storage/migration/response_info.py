from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ert.storage.local_storage import local_storage_get_ert_config

if TYPE_CHECKING:
    from pathlib import Path


def migrate(path: Path) -> None:
    ert_config = local_storage_get_ert_config()
    ens_config = ert_config.ensemble_config
    for experiment in path.glob("experiments/*"):
        response_info = {}
        for response in ens_config.response_configuration:
            response_info[response.name] = response.to_dict()
        with open(experiment / "responses.json", "w", encoding="utf-8") as fout:
            fout.write(json.dumps(response_info, default=str))
    with open(path / "index.json", encoding="utf-8") as f:
        index_json = json.load(f)
    index_json["version"] = 3
    with open(path / "index.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(index_json))
