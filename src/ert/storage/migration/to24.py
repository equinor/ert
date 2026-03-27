from pathlib import Path

import polars as pl

info = "Add default None values to RFT observations and responses"


def migrate(path: Path) -> None:
    for rft_obs in path.glob("experiments/*/observations/rft"):
        rft_obs_df = pl.read_parquet(rft_obs)

        if "zone" not in rft_obs_df.columns:
            rft_obs_df = rft_obs_df.with_columns(
                pl.lit(None, dtype=pl.String).alias("zone")
            )

        rft_obs_df.write_parquet(rft_obs)

    for rft_response in path.glob("ensembles/*/*/rft.parquet"):
        rft_response_df = pl.read_parquet(rft_response)

        if "zone" not in rft_response_df.columns:
            rft_response_df = rft_response_df.with_columns(
                pl.lit(None, dtype=pl.String).alias("zone")
            )
        rft_response_df.write_parquet(rft_response)
