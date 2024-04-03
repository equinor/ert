from __future__ import annotations

import os.path
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
                obs_names = set(dataset["obs_name"].data)
                for obs_name in obs_names:
                    if os.path.exists(output_path / obs_name):
                        os.remove(output_path / obs_name)

                dataset.to_netcdf(output_path / f"{name}", engine="scipy")
