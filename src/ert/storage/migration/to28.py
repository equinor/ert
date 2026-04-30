import json
import logging
from pathlib import Path
from typing import Any

import polars as pl

info = """Splits rft.parquet into 2 files. rft.parquet has response only data and
       rft_observation_location_metadata.parquet has location related data."""

logger = logging.getLogger(__name__)


def original_response_schema() -> dict[str, Any]:
    return {
        "realization": pl.UInt16,
        "response_key": pl.String,
        "well": pl.String,
        "date": pl.String,
        "property": pl.String,
        "time": pl.Date,
        "depth": pl.Float32,
        "values": pl.Float32,
        "zone": pl.String,
        "east": pl.Float32,
        "north": pl.Float32,
        "tvd": pl.Float32,
        "i": pl.Int64,
        "j": pl.Int64,
        "k": pl.Int64,
    }


def observation_schema() -> dict[str, Any]:
    return {
        "response_key": pl.String,
        "well": pl.String,
        "date": pl.String,
        "observation_key": pl.String,
        "east": pl.Float32,
        "north": pl.Float32,
        "tvd": pl.Float32,
        "md": pl.Float32,
        "zone": pl.String,
        "observations": pl.Float32,
        "std": pl.Float32,
        "radius": pl.Float32,
    }


def response_schema() -> dict[str, Any]:
    return {
        "realization": pl.UInt16,
        "response_key": pl.String,
        "well": pl.String,
        "date": pl.String,
        "property": pl.String,
        "time": pl.Date,
        "depth": pl.Float32,
        "values": pl.Float32,
        "well_connection_cell": pl.Array(pl.Int64, 3),
    }


def location_metadata_schema() -> dict[str, Any]:
    return {
        "east": pl.Float32,
        "north": pl.Float32,
        "tvd": pl.Float32,
        "actual_zones": pl.List(pl.String),
        "well_connection_cell": pl.Array(pl.Int64, 3),
    }


class InvalidSchemaError(Exception):
    pass


def check_well_connection_cell_consistency(
    df: pl.DataFrame, group_cols: list[str]
) -> None:
    consistency_df = (
        df.group_by(group_cols)
        .agg(pl.col("well_connection_cell").n_unique().alias("n_unique_cells"))
        .filter(pl.col("n_unique_cells") > 1)
    )

    error_msg = (
        f"Unexpected: found {group_cols} combinations with inconsistent "
        f"well_connection_cell values:\n{consistency_df}\n"
        f"Each {group_cols} should map to exactly one well_connection_cell."
    )
    if len(consistency_df) > 0:
        raise RuntimeError(error_msg)


def _extract_response_df(original_rft_response: pl.DataFrame) -> pl.DataFrame:
    rft_response_df = original_rft_response.select(
        [
            "realization",
            "response_key",
            "well",
            "date",
            "property",
            "time",
            "depth",
            "values",
            pl.concat_list(
                pl.col("i").cast(pl.Int64),
                pl.col("j").cast(pl.Int64),
                pl.col("k").cast(pl.Int64),
            )
            .cast(pl.Array(pl.Int64, 3))
            .alias("well_connection_cell"),
        ]
    ).unique(maintain_order=True)

    well_date_depth_cols = ["well", "date", "depth"]
    check_well_connection_cell_consistency(rft_response_df, well_date_depth_cols)
    return rft_response_df


def _extract_locations_in_response_df(
    original_rft_response: pl.DataFrame,
) -> pl.DataFrame:
    locations_in_response_df = original_rft_response.select(
        [
            pl.col("east").cast(pl.Float32),
            pl.col("north").cast(pl.Float32),
            pl.col("tvd").cast(pl.Float32),
            # actually simulated zone and well_connection_cell are lost,
            # so if they were not equal to expected, they will be set to [] or None
            pl.col("zone").alias("actual_zone"),
            pl.concat_list(
                pl.col("i").cast(pl.Int64),
                pl.col("j").cast(pl.Int64),
                pl.col("k").cast(pl.Int64),
            )
            .cast(pl.Array(pl.Int64, 3))
            .alias("well_connection_cell"),
        ]
    )

    locations_in_response_non_null_df = locations_in_response_df.filter(
        pl.col("east").is_not_null()
        | pl.col("north").is_not_null()
        | pl.col("tvd").is_not_null()
    )

    location_cols = ["east", "north", "tvd"]
    check_well_connection_cell_consistency(
        locations_in_response_non_null_df, location_cols
    )

    return locations_in_response_non_null_df.group_by(location_cols).agg(
        pl.col("actual_zone")
        .filter(pl.col("actual_zone").is_not_null())
        .unique()
        .alias("actual_zones"),
        pl.col("well_connection_cell").first().alias("well_connection_cell"),
    )


def transform(
    observations_df: pl.DataFrame,
    original_rft_response: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:

    if original_rft_response.schema != original_response_schema():
        raise InvalidSchemaError(
            f"Unexpected schema for rft.parquet: {original_rft_response.schema}. "
            f"Expected: {original_response_schema()}"
        )

    if observations_df.schema != observation_schema():
        raise InvalidSchemaError(
            f"Unexpected schema for observations: {observations_df.schema}. "
            f"Expected: {observation_schema()}"
        )

    rft_response_df = _extract_response_df(original_rft_response)

    if observations_df.is_empty():
        empty_location_metadata_df = pl.DataFrame([], schema=location_metadata_schema())
        return rft_response_df, empty_location_metadata_df

    locations_in_response_df = _extract_locations_in_response_df(original_rft_response)

    observed_locations_df = observations_df.select(
        [
            pl.col("east").cast(pl.Float32),
            pl.col("north").cast(pl.Float32),
            pl.col("tvd").cast(pl.Float32),
        ]
    ).unique(maintain_order=True)

    locations_metadata_df = (
        observed_locations_df.join(
            locations_in_response_df,
            on=["east", "north", "tvd"],
            how="left",
            coalesce=False,
        )
        .with_columns(
            pl.col("actual_zones").fill_null(pl.lit([], dtype=pl.List(pl.String)))
        )
        .select(
            [
                "east",
                "north",
                "tvd",
                "actual_zones",
                "well_connection_cell",
            ]
        )
        .cast(pl.Schema(location_metadata_schema()))
        .unique(maintain_order=True)
    )

    return rft_response_df, locations_metadata_df


def migrate(path: Path) -> None:
    for rft_response_path in path.glob("ensembles/*/*/rft.parquet"):
        ensemble_dir = rft_response_path.parent.parent
        ensemble_index = json.loads(
            (ensemble_dir / "index.json").read_text(encoding="utf-8")
        )
        experiment_id = ensemble_index["experiment_id"]

        rft_obs_path = path / "experiments" / experiment_id / "observations" / "rft"
        rft_obs_df = None
        if not rft_obs_path.exists():
            logger.warning(
                f"Unexpected situation: RFT response exists at {rft_response_path}, "
                f"but RFT observations are missing at {rft_obs_path}. "
                "Assuming empty observations",
                stacklevel=2,
            )
            rft_obs_df = pl.DataFrame(schema=observation_schema())
        else:
            rft_obs_df = pl.read_parquet(rft_obs_path)

        original_rft_response_df = pl.read_parquet(rft_response_path)

        try:
            rft_response_df, location_metadata_df = transform(
                rft_obs_df, original_rft_response_df
            )
            logger.info(f"Successfully transformed RFT response at {rft_response_path}")
        except InvalidSchemaError as e:
            logger.warning(
                f"Schema error on migrating {rft_response_path}: {e}. "
                "Using empty dataframes instead.",
                stacklevel=2,
            )
            rft_response_df = pl.DataFrame(schema=response_schema())
            location_metadata_df = pl.DataFrame(schema=location_metadata_schema())

        location_metadata_path = (
            rft_response_path.parent / "rft_observation_location_metadata.parquet"
        )
        rft_response_df.write_parquet(rft_response_path)
        location_metadata_df.write_parquet(location_metadata_path)
