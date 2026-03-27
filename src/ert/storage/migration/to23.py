from pathlib import Path

import polars as pl

info = "Add default None values to summary observations LOCATION keywords"

old_localization_keywords = ["location_x", "location_y", "location_range"]
new_localization_keywords = ["east", "north", "radius"]


def migrate(path: Path) -> None:
    for gen_obs in path.glob("experiments/*/observations/gen_data"):
        gen_obs_df = pl.read_parquet(gen_obs)

        for new_kw in new_localization_keywords:
            if new_kw not in gen_obs_df.columns:
                gen_obs_df = gen_obs_df.with_columns(
                    pl.lit(None, dtype=pl.Float32).alias(new_kw)
                )

        gen_obs_df.write_parquet(gen_obs)

    for rft_obs in path.glob("experiments/*/observations/rft"):
        rft_obs_df = pl.read_parquet(rft_obs)

        for new_kw in new_localization_keywords:
            if new_kw not in rft_obs_df.columns:
                rft_obs_df = rft_obs_df.with_columns(
                    pl.lit(None, dtype=pl.Float32).alias(new_kw)
                )

        rft_obs_df.write_parquet(rft_obs)

    for summary_obs in path.glob("experiments/*/observations/summary"):
        summary_df = pl.read_parquet(summary_obs)

        for old_kw, new_kw in zip(
            old_localization_keywords, new_localization_keywords, strict=True
        ):
            if old_kw in summary_df.columns:
                column = summary_df[old_kw]
                summary_df = summary_df.with_columns(column.alias(new_kw))
                summary_df = summary_df.drop(old_kw)
            else:
                summary_df = summary_df.with_columns(
                    pl.lit(None, dtype=pl.Float32).alias(new_kw)
                )

        summary_df.write_parquet(summary_obs)
