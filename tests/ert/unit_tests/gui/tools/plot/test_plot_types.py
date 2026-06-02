import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from ert.field_utils import AxisOrientation, ErtboxParameters
from ert.gui.tools.plot.plot_types import transform_observation_locations


@pytest.mark.parametrize(
    (
        "east",
        "north",
        "radius",
        "ertbox_params",
        "result_loc",
    ),
    [
        pytest.param(
            [1, 2],
            [0, 0],
            [100.0, 200.0],
            ErtboxParameters(
                origin=(1000.0, 1000.0),
                rotation_angle=0.0,
                nx=10,
                ny=10,
                nz=10,
                axis_orientation=AxisOrientation.LEFT_HANDED,
                xinc=100.0,
                yinc=200.0,
            ),
            None,
            id="Observations outside the field (left-handed)",
        ),
        pytest.param(
            [1.0, 101.0],
            [201.0, 201.0],
            [100.0, 200.0],
            ErtboxParameters(
                origin=(1.0, 1.0),
                rotation_angle=0.0,
                nx=10,
                ny=10,
                nz=10,
                axis_orientation=AxisOrientation.LEFT_HANDED,
                xinc=100.0,
                yinc=200.0,
            ),
            (
                np.array([0.0, 1.0]),
                np.array([1.0, 1.0]),
                np.array([1.0, 2.0]),
                np.array([0.5, 1.0]),
                np.array(["", ""]),
                np.array(["", ""]),
            ),
            id="Observations inside the field (left-handed)",
        ),
        pytest.param(
            [1.0, 101.0],
            [201.0, 201.0],
            [100.0, 200.0],
            ErtboxParameters(
                origin=(1.0, 1.0),
                rotation_angle=0.0,
                nx=10,
                ny=10,
                nz=10,
                axis_orientation=AxisOrientation.RIGHT_HANDED,
                xinc=100.0,
                yinc=200.0,
            ),
            (
                np.array([0.0, 1.0]),
                np.array([9.0, 9.0]),
                np.array([1.0, 2.0]),
                np.array([0.5, 1.0]),
                np.array(["", ""]),
                np.array(["", ""]),
            ),
            id="Observations inside the field (right-handed)",
        ),
    ],
)
def test_that_transform_observation_locations_handles_different_cases(
    east, north, radius, ertbox_params, result_loc
):
    df = pd.DataFrame({"east": east, "north": north, "radius": radius})
    result = transform_observation_locations(df, ertbox_params)
    if result is None:
        assert result_loc is None
    else:
        expected_x, expected_y, expected_radius_x, expected_radius_y, keys, indices = (
            result_loc
        )
        assert_allclose(result.x, expected_x)
        assert_allclose(result.y, expected_y)
        assert_allclose(result.radius_x, expected_radius_x)
        assert_allclose(result.radius_y, expected_radius_y)
        assert np.array_equal(result.observation_key, keys)
        assert np.array_equal(result.observation_index, indices)
