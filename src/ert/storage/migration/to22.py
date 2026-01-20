from pathlib import Path

import polars as pl

info = "Add default None values to summary observations LOCATION keywords"


def migrate(path: Path) -> None:
    for summary_observation in path.glob("experiments/*/observations/summary"):
        summary_df = pl.read_parquet(summary_observation)

        for location_kw in ["location_x", "location_y", "location_range"]:
            if location_kw not in summary_df.columns:
                summary_df = summary_df.with_columns(
                    pl.lit(None, dtype=pl.Float32).alias(location_kw)
                )

        summary_df.write_parquet(summary_observation)
