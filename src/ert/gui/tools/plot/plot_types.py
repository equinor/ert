from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd

from ert.field_utils import (
    AxisOrientation,
    ErtboxParameters,
    transform_positions_to_local_field_coordinates,
)


@dataclass(frozen=True)
class ObservationPlotLocations:
    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    radius_x: npt.NDArray[np.float64]
    radius_y: npt.NDArray[np.float64]
    observation_key: npt.NDArray[np.str_]
    observation_index: npt.NDArray[np.str_]


def transform_observation_locations(
    obs_loc_df: pd.DataFrame, ertbox_params: ErtboxParameters
) -> ObservationPlotLocations | None:
    """Return observation plot geometry and identity columns for localized obs."""
    if (
        ertbox_params.origin is not None
        and ertbox_params.rotation_angle is not None
        and ertbox_params.xinc is not None
        and ertbox_params.yinc is not None
        and not obs_loc_df.empty
    ):
        xpos, ypos = transform_positions_to_local_field_coordinates(
            ertbox_params.origin,
            ertbox_params.rotation_angle,
            obs_loc_df["east"].to_numpy(dtype=np.float64),
            obs_loc_df["north"].to_numpy(dtype=np.float64),
        )
        xpos /= ertbox_params.xinc
        ypos /= ertbox_params.yinc
        height, width = (
            ertbox_params.ny,
            ertbox_params.nx,
        )
        if ertbox_params.axis_orientation == AxisOrientation.RIGHT_HANDED:
            ypos = height - ypos

        radius = obs_loc_df["radius"].to_numpy(dtype=np.float64)
        radius_x = radius / ertbox_params.xinc
        radius_y = radius / ertbox_params.yinc

        inside_box = (
            np.isfinite(xpos)
            & np.isfinite(ypos)
            & np.isfinite(radius_x)
            & np.isfinite(radius_y)
            & (xpos >= 0)
            & (xpos < width)
            & (ypos >= 0)
            & (ypos < height)
        )
        if inside_box.any():
            observation_key = obs_loc_df.get(
                "observation_key", pd.Series("", index=obs_loc_df.index)
            ).to_numpy(dtype=str)
            observation_index = obs_loc_df.get(
                "observation_index", pd.Series("", index=obs_loc_df.index)
            ).to_numpy(dtype=str)
            return ObservationPlotLocations(
                x=xpos[inside_box],
                y=ypos[inside_box],
                radius_x=radius_x[inside_box],
                radius_y=radius_y[inside_box],
                observation_key=observation_key[inside_box],
                observation_index=observation_index[inside_box],
            )
    return None


LocalizationProvider: TypeAlias = Callable[
    [str, str, str, str], npt.NDArray[np.float32] | None
]
