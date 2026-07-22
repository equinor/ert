from typing import Any

import polars as pl

from ert.config._shapes import PolygonShapeConfig, ShapeRegistry
from ert.config.rft_config import RFTConfig


def ensure_qc_error_column(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure the DataFrame has a 'qc_error' column, initializing it to null if not
    present.
    """
    if "qc_error" not in df.columns:
        return df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("qc_error"))
    return df


def append_to_qc_error(condition: pl.Expr, error_message: pl.Expr) -> pl.Expr:
    """Append error_message to qc_error if condition."""
    return (
        pl.when(condition)
        .then(
            pl.when(pl.col("qc_error").is_not_null())
            .then(
                pl.concat_str(
                    pl.col("qc_error"),
                    pl.lit(";\n"),
                    error_message,
                )
            )
            .otherwise(error_message)
        )
        .otherwise(pl.col("qc_error"))
    )


def _qc_observation_zone_matches_simulated_zones(
    df: pl.DataFrame,
) -> pl.DataFrame:
    zone_error = pl.concat_str(
        pl.lit("expected zone '"),
        pl.col("expected_zone"),
        pl.lit("' did not match any of the simulated zones: "),
        pl.col("actual_zones").list.join(", "),
    )

    return df.with_columns(
        append_to_qc_error(~RFTConfig.is_zone_valid(), zone_error).alias("qc_error")
    )


def _qc_observation_location_is_in_the_grid(
    df: pl.DataFrame,
) -> pl.DataFrame:
    location_error = pl.concat_str(
        pl.lit("did not find grid coordinate for location "),
        pl.col("east").cast(pl.String),
        pl.lit(", "),
        pl.col("north").cast(pl.String),
        pl.lit(", "),
        pl.col("tvd").cast(pl.String),
    )

    return df.with_columns(
        append_to_qc_error(
            pl.col("well_connection_cell").is_null(), location_error
        ).alias("qc_error")
    )


def qc_rft_observations(
    observations_with_metadata: pl.DataFrame,
) -> pl.DataFrame:
    """
    Perform quality control checks on RFT observations combined with their response
    metadata.
    """
    qc_df = ensure_qc_error_column(observations_with_metadata)
    qc_df = _qc_observation_zone_matches_simulated_zones(qc_df)
    return _qc_observation_location_is_in_the_grid(qc_df)


def _qc_observation_location_is_inside_boundary(
    df: pl.DataFrame, shape_registry: ShapeRegistry
) -> pl.DataFrame:
    """Remove observations that are outside the limiting observation boundary.

    Some observations do not belong to the provided polygon boundary. This is not
    considered an error, they are just silently removed.
    """

    def is_inside_boundary(row: dict[str, Any]) -> bool:
        boundary_id = row["boundary_id"]
        if boundary_id is None:
            return True
        boundary = shape_registry.get(boundary_id)
        assert boundary is not None, (
            f"Boundary with ID {boundary_id} not found in shape registry."
        )
        assert isinstance(boundary, PolygonShapeConfig)
        return boundary.contains(row["east"], row["north"])

    return df.filter(
        pl.struct(["boundary_id", "east", "north"]).map_elements(
            is_inside_boundary, return_dtype=pl.Boolean
        )
    )


def qc_seismic_observations(
    observations: pl.DataFrame,
    shape_registry: ShapeRegistry,
) -> pl.DataFrame:
    """Perform quality control checks on seismic observations."""
    return _qc_observation_location_is_inside_boundary(observations, shape_registry)
