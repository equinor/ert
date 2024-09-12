import dataclasses
import json
import os
from pathlib import Path
from typing import List, Optional

import polars
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


@dataclasses.dataclass
class ObservationDatasetInfo:
    polars_df: polars.DataFrame
    response_type: str
    original_ds_path: Path

    @classmethod
    def from_path(cls, path: Path) -> "ObservationDatasetInfo":
        observation_key = os.path.basename(path)
        ds = xr.open_dataset(path, engine="scipy")
        response_key = ds.attrs["response"]
        response_type = "summary" if response_key == "summary" else "gen_data"

        df = polars.from_pandas(ds.to_dataframe().dropna().reset_index())
        df = df.with_columns(observation_key=polars.lit(observation_key))

        primary_key = (
            ["time"] if response_type == "summary" else ["report_step", "index"]
        )
        if response_type == "summary":
            df = df.rename({"name": "response_key"})
            df = df.with_columns(polars.col("time").dt.cast_time_unit("ms"))

        if response_type == "gen_data":
            df = df.with_columns(
                polars.col("report_step").cast(polars.UInt16),
                polars.col("index").cast(polars.UInt16),
                response_key=polars.lit(response_key),
            )

        df = df.with_columns(
            [
                polars.col("std").cast(polars.Float32),
                polars.col("observations").cast(polars.Float32),
            ]
        )

        df = df[
            ["response_key", "observation_key", *primary_key, "observations", "std"]
        ]

        return cls(df, response_type, path)


def _migrate_observations_to_grouped_parquet(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        if not os.path.exists(experiment / "observations"):
            os.makedirs(experiment / "observations")

        _obs_keys = os.listdir(os.path.join(experiment, "observations"))

        if len(set(_obs_keys) - {"summary", "gen_data"}) == 0:
            # Observations are already migrated, likely from .to4 migrations
            continue

        obs_ds_infos = [
            ObservationDatasetInfo.from_path(experiment / "observations" / p)
            for p in _obs_keys
        ]

        for response_type in ["gen_data", "summary"]:
            infos = [
                _info for _info in obs_ds_infos if _info.response_type == response_type
            ]
            if len(infos) > 0:
                concatd_df = polars.concat([_info.polars_df for _info in infos])
                concatd_df.write_parquet(experiment / "observations" / response_type)

                for _info in infos:
                    os.remove(_info.original_ds_path)


def _migrate_responses_from_netcdf_to_parquet(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        ensembles = path.glob("ensembles/*")

        experiment_id = None
        with open(experiment / "index.json", encoding="utf-8") as f:
            exp_index = json.load(f)
            experiment_id = exp_index["id"]

        for ens in ensembles:
            with open(ens / "index.json", encoding="utf-8") as f:
                ens_file = json.load(f)
                if ens_file["experiment_id"] != experiment_id:
                    continue

            real_dirs = [*ens.glob("realization-*")]

            for real_dir in real_dirs:
                for ds_name in ["gen_data", "summary"]:
                    if (real_dir / f"{ds_name}.nc").exists():
                        gen_data_ds = xr.open_dataset(
                            real_dir / f"{ds_name}.nc", engine="scipy"
                        )

                        pandas_df = gen_data_ds.to_dataframe().dropna()
                        polars_df = polars.from_pandas(pandas_df.reset_index())
                        polars_df = polars_df.rename({"name": "response_key"})

                        if "time" in polars_df:
                            polars_df = polars_df.with_columns(
                                polars.col("time").dt.cast_time_unit("ms")
                            )

                        # Ensure "response_key" is the first column
                        polars_df = polars_df.select(
                            ["response_key"]
                            + [
                                col
                                for col in polars_df.columns
                                if col != "response_key"
                            ]
                        )
                        polars_df.write_parquet(real_dir / f"{ds_name}.parquet")

                        os.remove(real_dir / f"{ds_name}.nc")


def migrate(path: Path) -> None:
    _migrate_response_datasets(path)
    _migrate_response_configs(path)
    _migrate_responses_from_netcdf_to_parquet(path)
    _migrate_observations_to_grouped_parquet(path)
