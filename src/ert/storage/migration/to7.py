import json
import os
from pathlib import Path
from typing import List, Optional

import xarray as xr

info = "Standardize response configs"


def _migrate_response_configs(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        with open(experiment / "responses.json", encoding="utf-8") as fin:
            responses = json.load(fin)

        # If we for example do a .to2() migration
        # this will implicitly upgrade it to v7 since
        # it loads an ensemble config which again writes a response config
        is_already_migrated = (
            "summary" in responses and "input_files" in responses["summary"]
        ) or "gen_data" in responses

        if is_already_migrated:
            return

        migrated = {}
        if "summary" in responses:
            original = responses["summary"]
            migrated["summary"] = {
                "_ert_kind": "SummaryConfig",
                "name": "summary",
                "input_files": [original["input_file"]],
                "keys": sorted(set(original["keys"])),
            }

        gendata_responses = {
            k: v for k, v in responses.items() if v["_ert_kind"] == "GenDataConfig"
        }

        if gendata_responses:
            migrated["gen_data"] = {
                "_ert_kind": "GenDataConfig",
            }

            keys = []
            input_files = []
            report_steps = []

            for name, info in gendata_responses.items():
                keys.append(name)
                report_steps.append(info["report_steps"])
                input_files.append(info["input_file"])

            migrated["gen_data"].update(
                {
                    "keys": keys,
                    "input_files": input_files,
                    "report_steps_list": report_steps,
                }
            )

        with open(experiment / "responses.json", "w", encoding="utf-8") as fout:
            json.dump(migrated, fout)


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
                        xr.open_dataset(
                            real_dir / f"{gendata_name}.nc", engine="scipy"
                        ).expand_dims(name=[gendata_name], axis=1),
                    )
                    for gendata_name in gendata_keys
                    if os.path.exists(real_dir / f"{gendata_name}.nc")
                ]

                if gen_data_datasets:
                    gen_data_combined = _ensure_coord_order(
                        xr.concat([ds for _, ds in gen_data_datasets], dim="name")
                    )
                    gen_data_combined.to_netcdf(
                        real_dir / "gen_data.nc", engine="scipy"
                    )

                    for p in [ds_path for ds_path, _ in gen_data_datasets]:
                        os.remove(p)


def migrate(path: Path) -> None:
    _migrate_response_datasets(path)
    _migrate_response_configs(path)
