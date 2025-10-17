# Jonak, should we convert all .nc files, or just look in the experiment/parameters.json
#        file for what is a everest_parameters
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
    current_df = None
    for netcdf_file in path.glob("*.nc"):
        ds = xr.open_dataset(netcdf_file)
        df: pd.DataFrame = ds["values"].to_pandas()
        df = df.reset_index()

        # Convert pandas DataFrame to polars DataFrame
        pl_df = pl.from_pandas(df)
        pl_df = pl_df.rename({"realizations": "realization"})
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
        current_df = pl.concat([current_df, old_df])

    current_df.write_parquet(path / ".." / "SCALAR.parquet")


def migrate(path):
    migrate_netcdf_content_to_scalars(path)
