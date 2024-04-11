from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ert.storage.local_storage import local_storage_get_ert_config

if TYPE_CHECKING:
    from pathlib import Path


info = "Introducing observations and removing template_file_path"


def migrate(path: Path) -> None:
    ert_config = local_storage_get_ert_config()
    observations = ert_config.observations.datasets

    for experiment in path.glob("experiments/*"):
        if observations:
            output_path = experiment / "observations"
            output_path.mkdir(parents=True, exist_ok=True)

            for response_type, multi_obs_ds in observations.items():
                for obs_name, single_obs_ds in multi_obs_ds.groupby(
                    "obs_name", squeeze=True
                ):
                    if response_type == "summary":
                        single_obs_ds.drop("obs_name").to_netcdf(
                            output_path / obs_name, engine="scipy"
                        )
                    else:
                        squeezed = single_obs_ds.drop(["obs_name", "name"]).squeeze(
                            "name", drop=True
                        )
                        squeezed.attrs["response"] = obs_name
                        squeezed.to_netcdf(output_path / obs_name, engine="scipy")

        with open(experiment / "parameter.json", encoding="utf-8") as fin:
            parameters_json = json.load(fin)
        with open(experiment / "parameter.json", "w", encoding="utf-8") as fout:
            fout.write(json.dumps(parameters_json, indent=3))
