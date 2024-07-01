import json
import os
from pathlib import Path
from typing import List, Optional

import xarray as xr

from ert.storage.local_ensemble import RealizationState

info = (
    "Combining datasets for responses and parameters."
    "Rename and change transfer_function_definitions"
)


def _ensure_coord_order(
    ds: xr.Dataset, dim_order_explicit: Optional[List[str]] = None
) -> xr.Dataset:
    # Copypaste'd from LocalEnsemble
    data_vars = list(ds.data_vars.keys())

    # We assume only data vars with the same dimensions,
    # i.e., (realization, *index) for all of them.
    dim_order_of_first_var = (
        ds[data_vars[0]].dims if dim_order_explicit is None else dim_order_explicit
    )
    return ds[[*dim_order_of_first_var, *data_vars]].sortby(
        dim_order_of_first_var[0]  # "realization" / "realizations"
    )


def migrate(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        with open(experiment / "parameter.json", encoding="utf-8") as fin:
            parameters_json = json.load(fin)

        with open(experiment / "parameter.json", "w", encoding="utf-8") as fout:
            for param in parameters_json.values():
                if "transfer_function_definitions" in param:
                    param["transform_function_definitions"] = param[
                        "transfer_function_definitions"
                    ]
                    del param["transfer_function_definitions"]

                if "transform_function_definitions" in param:
                    transform_function_definitions = []
                    for tfd in param["transform_function_definitions"]:
                        if isinstance(tfd, str):
                            items = tfd.split()
                            transform_function_definitions.append(
                                {
                                    "name": items[0],
                                    "param_name": items[1],
                                    "values": items[2:],
                                }
                            )
                        elif isinstance(tfd, dict):
                            transform_function_definitions.append(tfd)

                    param["transform_function_definitions"] = (
                        transform_function_definitions
                    )
            fout.write(json.dumps(parameters_json, indent=4))

    for experiment in path.glob("experiments/*"):
        ensembles = path.glob("ensembles/*")

        experiment_id = None
        with open(experiment / "index.json") as f:
            exp_index = json.load(f)
            experiment_id = exp_index["id"]

        responses_file = experiment / "responses.json"
        with open(responses_file, encoding="utf-8", mode="r") as f:
            responses_obj = json.load(f)

        assert (
            responses_obj is not None
        ), f"Failed to load responses.json @ {responses_file}"

        gen_data_keys = {
            k for k, v in responses_obj.items() if v["_ert_kind"] == "GenDataConfig"
        }

        params_file = experiment / "parameter.json"
        with open(params_file, encoding="utf-8", mode="r") as f:
            params_obj = json.load(f)

        assert params_obj is not None, f"Failed to load parameters.json @ {params_file}"
        all_param_groups = set(params_obj.keys())

        for ens in ensembles:
            with open(ens / "index.json") as f:
                ens_file = json.load(f)
                if ens_file["experiment_id"] != experiment_id:
                    continue

            real_dirs = [*ens.glob("realization-*")]

            # Combine summaries
            summary_datasets = [
                (p / "summary.nc", xr.open_dataset(p / "summary.nc"))
                for p in real_dirs
                if os.path.exists(p / "summary.nc")
            ]

            # Combine responses, for every response name
            gen_data_datasets = []
            for response_name in gen_data_keys:
                gen_data_datasets.extend(
                    [
                        (
                            p / f"{response_name}.nc",
                            xr.open_dataset(p / f"{response_name}.nc").expand_dims(
                                name=[response_name], axis=1
                            ),
                        )
                        for p in real_dirs
                        if os.path.exists(p / f"{response_name}.nc")
                    ]
                )

            # Combine parameters by name within group
            param_datasets = []
            for param_group in all_param_groups:
                datasets_for_group = [
                    (p / f"{param_group}.nc", xr.open_dataset(p / f"{param_group}.nc"))
                    for p in real_dirs
                    if os.path.exists(p / f"{param_group}.nc")
                ]

                param_datasets.append((param_group, datasets_for_group))

            state_maps = RealizationState()

            for i in range(ens_file["ensemble_size"]):
                real_dir = ens / f"realization-{i}"
                for _param_group, _datasets in param_datasets:
                    state_maps.add(
                        i,
                        {
                            (
                                _param_group,
                                _param_group,
                                os.path.exists(real_dir / f"{_param_group}.nc"),
                            )
                        },
                    )

                for _key in gen_data_keys:
                    state_maps.add(
                        i, {("gen_data", _key, os.path.exists(real_dir / f"{_key}.nc"))}
                    )
                state_maps.add(
                    i,
                    {
                        (
                            "summary",
                            "summary",
                            os.path.exists(os.path.exists(real_dir / "summary.nc")),
                        )
                    },
                )

            state_maps.to_file(ens / "state_map.json")

            if gen_data_datasets:
                gen_data_combined = _ensure_coord_order(
                    xr.concat([ds for _, ds in gen_data_datasets], dim="realization")
                )
                gen_data_combined.to_netcdf(ens / "gen_data.nc", engine="scipy")

                for p in [ds_path for ds_path, _ in gen_data_datasets]:
                    os.remove(p)

            if summary_datasets:
                summary_combined = _ensure_coord_order(
                    xr.concat([ds for _, ds in summary_datasets], dim="realization")
                )
                summary_combined.to_netcdf(ens / "summary.nc", engine="scipy")
                for p in [ds_path for ds_path, _ in summary_datasets]:
                    os.remove(p)

            if param_datasets:
                for param_group, datasets in param_datasets:
                    if len(datasets) > 0:
                        params_combined = _ensure_coord_order(
                            xr.concat([ds for _, ds in datasets], dim="realizations")
                        )
                        params_combined.to_netcdf(
                            ens / f"{param_group}.nc", engine="scipy"
                        )
                        for p in [ds_path for ds_path, _ in datasets]:
                            os.remove(p)

            for p in real_dirs:
                if not os.listdir(p):
                    os.rmdir(p)

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
            combined.to_netcdf(
                experiment / "observations" / "summary.nc", engine="scipy"
            )

        if len(gen_obs) > 0:
            combined = _ensure_coord_order(
                xr.concat(gen_obs, dim="obs_name").transpose(
                    "name", "obs_name", "index", "report_step"
                )
            )
            combined.attrs["response"] = "gen_data"
            combined.to_netcdf(
                experiment / "observations" / "gen_data.nc", engine="scipy"
            )

        for single_obs_file in to_remove:
            os.remove(single_obs_file)
