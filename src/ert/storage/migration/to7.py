import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import xarray as xr

info = "Standardize response configs"


def _migrate_response_configs(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        with open(experiment / "responses.json", encoding="utf-8") as fin:
            responses = json.load(fin)

        standardized = {}

        if "summary" in responses:
            original = responses["summary"]

            smry_kwargs = {}
            if "refcase" in original:
                smry_kwargs["refcase"] = original["refcase"]

            standardized["summary"] = {
                "_ert_kind": "SummaryConfig",
                "args_per_instance": [
                    {
                        "name": "summary",
                        "input_file": original["input_file"],
                        "keys": sorted(
                            set(original["keys"])
                        ),  # Note: Maybe a bit more of a "cleanup" than migrate,
                        # but sometimes there are duplicate keys in configs
                        # ref the storage migration tests
                        "kwargs": smry_kwargs,
                    }
                ],
            }

        gendata_responses = {
            k: v for k, v in responses.items() if v["_ert_kind"] == "GenDataConfig"
        }

        if gendata_responses:
            args_per_instance: List[Dict[str, Any]] = []
            standardized["gen_data"] = {
                "_ert_kind": "GenDataConfig",
                "args_per_instance": args_per_instance,
            }
            for name, info in gendata_responses.items():
                instance_kwargs: Dict[str, Any] = {}
                instance_args = {
                    "name": name,
                    "input_file": info["input_file"],
                    "kwargs": instance_kwargs,
                }

                if "report_steps" in info:
                    instance_kwargs["report_steps"] = info["report_steps"]

                if "index" in info:
                    instance_kwargs["index"] = info["index"]

                if instance_kwargs:
                    instance_args["kwargs"] = instance_kwargs

                instance_args["keys"] = [name]

                args_per_instance.append(instance_args)

        with open(experiment / "responses.json", "w", encoding="utf-8") as fout:
            json.dump(standardized, fout)


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


def _migrate_response_datasets(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        ensembles = path.glob("ensembles/*")

        experiment_id = None
        with open(experiment / "index.json", encoding="utf-8") as f:
            exp_index = json.load(f)
            experiment_id = exp_index["id"]

        responses_file = experiment / "responses.json"
        with open(responses_file, encoding="utf-8", mode="r") as f:
            responses_obj = json.load(f)

        assert (
            responses_obj is not None
        ), f"Failed to load responses.json @ {responses_file}"

        gendata_keys = {
            k for k, v in responses_obj.items() if v["_ert_kind"] == "GenDataConfig"
        }

        for ens in ensembles:
            with open(ens / "index.json", encoding="utf-8") as f:
                ens_file = json.load(f)
                if ens_file["experiment_id"] != experiment_id:
                    continue

            real_dirs = [*ens.glob("realization-*")]

            for real_dir in real_dirs:
                # Combine responses, for every response name
                gen_data_datasets = [
                    (
                        real_dir / f"{gendata_name}.nc",
                        xr.open_dataset(real_dir / f"{gendata_name}.nc").expand_dims(
                            name=[gendata_name], axis=1
                        ),
                    )
                    for gendata_name in gendata_keys
                    if os.path.exists(real_dir / f"{gendata_name}.nc")
                ]

                if gen_data_datasets:
                    gen_data_combined = _ensure_coord_order(
                        xr.concat([ds for _, ds in gen_data_datasets], dim="name")
                    )
                    gen_data_combined.to_netcdf(real_dir / "gen_data.nc")

                    for p in [ds_path for ds_path, _ in gen_data_datasets]:
                        os.remove(p)


def migrate(path: Path) -> None:
    _migrate_response_datasets(path)
    _migrate_response_configs(path)
