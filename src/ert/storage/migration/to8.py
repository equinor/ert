import dataclasses
import json
import os
from pathlib import Path

import polars as pl
import xarray as xr

info = "Store observations & responses as parquet"


@dataclasses.dataclass
class ObservationDatasetInfo:
    polars_df: pl.DataFrame
    response_type: str
    original_ds_path: Path

    @classmethod
    def from_path(cls, path: Path) -> "ObservationDatasetInfo":
        observation_key = os.path.basename(path)
        ds = xr.open_dataset(path, engine="scipy")
        response_key = ds.attrs["response"]
        response_type = "summary" if response_key == "summary" else "gen_data"

        df = pl.from_pandas(
            ds.to_dataframe().dropna().reset_index(),
            schema_overrides={
                "report_step": pl.UInt16,
                "index": pl.UInt16,
                "observations": pl.Float32,
                "std": pl.Float32,
            }
            if response_type == "gen_data"
            else {
                "time": pl.Datetime("ms"),  # type: ignore
                "observations": pl.Float32,
                "std": pl.Float32,
            },
        )

        df = df.with_columns(observation_key=pl.lit(observation_key))

        primary_key = (
            ["time"] if response_type == "summary" else ["report_step", "index"]
        )
        if response_type == "summary":
            df = df.rename({"name": "response_key"})

        if response_type == "gen_data":
            df = df.with_columns(
                response_key=pl.lit(response_key),
            )

        df = df[
            ["response_key", "observation_key", *primary_key, "observations", "std"]
        ]

        return cls(df, response_type, path)


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
                for response_type, schema_overrides in [
                    (
                        "gen_data",
                        {
                            "realization": pl.UInt16,
                            "report_step": pl.UInt16,
                            "index": pl.UInt16,
                            "values": pl.Float32,
                        },
                    ),
                    (
                        "summary",
                        {
                            "realization": pl.UInt16,
                            "time": pl.Datetime("ms"),
                            "values": pl.Float32,
                        },
                    ),
                ]:
                    if (real_dir / f"{response_type}.nc").exists():
                        xr_ds = xr.open_dataset(
                            real_dir / f"{response_type}.nc",
                            engine="scipy",
                        )

                        pandas_df = xr_ds.to_dataframe().dropna().reset_index()
                        polars_df = pl.from_pandas(
                            pandas_df,
                            schema_overrides=schema_overrides,  # type: ignore
                        )
                        polars_df = polars_df.rename({"name": "response_key"})

                        # Ensure "response_key" is the first column
                        polars_df = polars_df.select(
                            ["response_key"]
                            + [
                                col
                                for col in polars_df.columns
                                if col != "response_key"
                            ]
                        )
                        polars_df.write_parquet(real_dir / f"{response_type}.parquet")

                        os.remove(real_dir / f"{response_type}.nc")


def _migrate_observations_to_grouped_parquet(path: Path) -> None:
    for experiment in path.glob("experiments/*"):
        if not os.path.exists(experiment / "observations"):
            os.makedirs(experiment / "observations")

        obs_keys = os.listdir(os.path.join(experiment, "observations"))

        if len(set(obs_keys) - {"summary", "gen_data"}) == 0:
            # Observations are already migrated, likely from .to4 migrations
            continue

        obs_ds_infos = [
            ObservationDatasetInfo.from_path(experiment / "observations" / p)
            for p in obs_keys
        ]

        for response_type in ["gen_data", "summary"]:
            infos = [
                _info for _info in obs_ds_infos if _info.response_type == response_type
            ]
            if len(infos) > 0:
                concatd_df = pl.concat([_info.polars_df for _info in infos])
                concatd_df.write_parquet(experiment / "observations" / response_type)

                for _info in infos:
                    os.remove(_info.original_ds_path)


def migrate(path: Path) -> None:
    _migrate_responses_from_netcdf_to_parquet(path)
    _migrate_observations_to_grouped_parquet(path)
