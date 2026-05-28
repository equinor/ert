import numpy as np
import pandas as pd
import pytest

from ert.field_utils import AxisOrientation, ErtboxParameters
from ert.gui.tools.plot.observation_locations import transform_observation_locations


@pytest.mark.parametrize(
    (
        "east",
        "north",
        "ertbox_params",
        "result_loc",
    ),
    [
        pytest.param(
            [1, 2],
            [0, 0],
            ErtboxParameters(
                origin=(1000.0, 1000.0),
                rotation_angle=0.0,
                nx=10,
                ny=10,
                nz=10,
                axis_orientation=AxisOrientation.LEFT_HANDED,
            ),
            None,
            id="Observations outside the field (left-handed)",
        ),
        pytest.param(
            [1, 2],
            [2, 2],
            ErtboxParameters(
                origin=(1.0, 1.0),
                rotation_angle=0.0,
                nx=10,
                ny=10,
                nz=10,
                axis_orientation=AxisOrientation.LEFT_HANDED,
            ),
            (np.array([0.0, 1.0], dtype=np.float32), np.array([1.0, 1.0])),
            id="Observations inside the field (left-handed)",
        ),
        pytest.param(
            [1, 2],
            [2, 2],
            ErtboxParameters(
                origin=(1.0, 1.0),
                rotation_angle=0.0,
                nx=10,
                ny=10,
                nz=10,
                axis_orientation=AxisOrientation.RIGHT_HANDED,
            ),
            (np.array([0.0, 1.0], dtype=np.float32), np.array([9.0, 9.0])),
            id="Observations inside the field (right-handed)",
        ),
    ],
)
def test_that_transform_observation_locations_handles_different_cases(
    east, north, ertbox_params, result_loc
):
    df = pd.DataFrame({"east": east, "north": north})
    result = transform_observation_locations(df, ertbox_params)
    if result is None:
        assert result_loc is None
    else:
        assert result_loc is not None
        expected_x, expected_y = result_loc
        assert np.array_equal(result.x, expected_x)
        assert np.array_equal(result.y, expected_y)
