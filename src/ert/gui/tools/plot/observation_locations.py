from __future__ import annotations

import numpy as np
import pandas as pd

from ert.field_utils import (
    AxisOrientation,
    ErtboxParameters,
    transform_positions_to_local_field_coordinates,
)

from .plot_types import ObservationPlotLocations


def transform_observation_locations(
    obs_loc_df: pd.DataFrame, ertbox_params: ErtboxParameters
) -> ObservationPlotLocations | None:
    if (
        ertbox_params.origin is not None
        and ertbox_params.rotation_angle is not None
        and not obs_loc_df.empty
    ):
        xpos, ypos = transform_positions_to_local_field_coordinates(
            ertbox_params.origin,
            ertbox_params.rotation_angle,
            obs_loc_df["east"].to_numpy(dtype=np.float64),
            obs_loc_df["north"].to_numpy(dtype=np.float64),
        )
        height, width = (
            ertbox_params.ny,
            ertbox_params.nx,
        )
        if ertbox_params.axis_orientation == AxisOrientation.RIGHT_HANDED:
            ypos = height - ypos

        inside_box = (
            np.isfinite(xpos)
            & np.isfinite(ypos)
            & (xpos >= 0)
            & (xpos < width)
            & (ypos >= 0)
            & (ypos < height)
        )
        if inside_box.any():
            return ObservationPlotLocations(
                x=xpos[inside_box].astype(np.float32),
                y=ypos[inside_box].astype(np.float32),
                radius_x=np.ones(inside_box.sum(), dtype=np.float64),
                radius_y=np.ones(inside_box.sum(), dtype=np.float64),
                observation_key=np.full(inside_box.sum(), "", dtype=str),
                observation_index=np.full(inside_box.sum(), "", dtype=str),
            )
    return None
