from pathlib import Path

import pandas as pd
import polars as pl
import xarray as xr

info = "Migrate control config from dataset in netcdf files to SCALARS.parquet"


def migrate_netcdf_content_to_scalars(path: Path) -> None:
    for ensemble_path in path.glob("ensembles/*"):
        for realization in ensemble_path.glob("realization-*"):
            _add_netcdf_content_to_scalars_control_values_to_scalars_parquet(
                path / ensemble_path / realization
            )


def _add_netcdf_content_to_scalars_control_values_to_scalars_parquet(
    path: Path,
) -> None:
    scalars_path = path / ".." / "SCALAR.parquet"
    current_df: pl.DataFrame | None = None
    realization = int(path.parts[-1].split("realization-")[1])
    for netcdf_file in path.glob("*.nc"):
        group_name = str(netcdf_file.parts[-1]).split(".nc")[0]
        ds = xr.open_dataset(netcdf_file)
        if len(ds.dims) > 2:
            continue  # Cannot convert to dataframe
        print(f"{ds=}")
        if "realizations" not in ds.dims:
            # realization is not guaranteed to be a dimension, so
            # we add it with the realization from path
            ds = ds.expand_dims(realizations=[realization])
        df: pd.DataFrame = ds["values"].to_pandas()
        df = df.reset_index()

        # Convert pandas DataFrame to polars DataFrame
        pl_df = pl.from_pandas(df)

        pl_df = pl_df.rename(
            {
                c: "{}.{}".format(group_name, c.replace("\0", "."))
                for c in pl_df.columns
                if c != "realizations"
            }
            | {"realizations": "realization"},
            strict=False,
        )
        if current_df is None:
            current_df = pl_df
        else:
            current_df = (
                current_df.join(pl_df, on="realization", how="left")
                .unique(subset=["realization"], keep="first")
                .sort("realization")
            )
    if scalars_path.exists():
        old_df = pl.read_parquet(scalars_path)
        if current_df is not None:
            current_df = pl.concat([current_df, old_df], how="vertical")
        else:
            current_df = old_df
    if current_df is not None:
        current_df = current_df.sort(pl.col("realization"))
        current_df.write_parquet(path / ".." / "SCALAR.parquet")


def migrate(path: Path) -> None:
    migrate_netcdf_content_to_scalars(path)
