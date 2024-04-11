import json
import os
from pathlib import Path

import xarray as xr

info = "Combining datasets for responses and parameters"


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        responses_file = experiment / "responses.json"

        with open(responses_file, encoding="utf-8", mode="r") as f:
            responses_obj = json.load(f)

        assert (
            responses_obj is not None
        ), f"Failed to load responses.json @ {responses_file}"

        gen_data_keys = {
            k for k, v in responses_obj.items() if v["_ert_kind"] == "GenDataConfig"
        }

        summary_obs = []
        gen_obs = []
        to_remove = []

        for single_obs_file in (experiment / "observations").glob("*"):
            obs_name = single_obs_file.name

            if obs_name.startswith("."):
                continue

            single_obs_ds = xr.open_dataset(single_obs_file)
            response = single_obs_ds.attrs["response"]

            if response == "summary":
                name = single_obs_ds["name"].data.tolist()
                assert len(name) == 1

                name = name[0]

                # expand_dims with axis=1
                # does not seem to properly reorder axes when
                # .to_dataframe() is called.
                # likely gone in newer xarray version.
                summary_obs.append(
                    single_obs_ds.squeeze("name").expand_dims(
                        name=[name], obs_name=[obs_name]
                    )
                )
                to_remove.append(single_obs_file)
            elif response in gen_data_keys:
                gen_obs.append(
                    single_obs_ds.expand_dims(name=[response], obs_name=[obs_name])
                )
                to_remove.append(single_obs_file)

        if len(summary_obs) > 0:
            combined = xr.concat(summary_obs, dim="obs_name")
            combined.attrs["response"] = "summary"
            combined.to_netcdf(experiment / "observations" / "summary", engine="scipy")

        if len(gen_obs) > 0:
            combined = xr.concat(gen_obs, dim="obs_name")
            combined.attrs["response"] = "gen_data"
            combined.to_netcdf(experiment / "observations" / "gen_data", engine="scipy")

        for single_obs_file in to_remove:
            os.remove(single_obs_file)
