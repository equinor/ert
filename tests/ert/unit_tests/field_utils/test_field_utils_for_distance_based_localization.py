import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from ert.field_utils import (
    AxisOrientation,
    ErtboxParameters,
    calc_rho_for_2d_grid_layer,
    transform_local_ellipse_angle_to_local_coords,
    transform_observation_locations,
    transform_positions_to_local_field_coordinates,
)


@pytest.mark.parametrize(
    ("coordsys_rotation", "ellipse_anisotropy_angle", "expected"),
    [
        (0.0, [45.0, -165.0], [45.0, -165.0]),
        (120.0, [-135.0, 175.0], [-255.0, 55.0]),
    ],
)
def test_transform_local_ellipse_angle_to_local_coords(
    coordsys_rotation,
    ellipse_anisotropy_angle,
    expected,
):
    angle_inputs = np.array(ellipse_anisotropy_angle)
    transf_angle = transform_local_ellipse_angle_to_local_coords(
        coordsys_rotation, angle_inputs
    )
    assert_allclose(transf_angle, expected)


@pytest.mark.parametrize(
    (
        "coordsys_origin",
        "coordsys_rotation",
        "utmx",
        "utmy",
        "expected_x",
        "expected_y",
    ),
    [
        (
            (0.0, 0.0),
            0.0,
            [1000.0, 100.0, -1000.0],
            [0.0, 500.0, 250.0],
            [1000.0, 100.0, -1000.0],
            [0.0, 500.0, 250.0],
        ),
        (
            (1000.0, 1500.0),
            0.0,
            [1000.0, 100.0, -1000.0],
            [0.0, 500.0, 2100.0],
            [0.0, -900.0, -2000.0],
            [-1500.0, -1000.0, 600.0],
        ),
        (
            (1000.0, 1500.0),
            60.0,
            [1000.0, 100.0, -1000.0],
            [0.0, 500.0, 2100.0],
            [-1299.03810568, -1316.02540378, -480.38475773],
            [-750.0, 279.42286341, 2032.05080757],
        ),
        (
            (1000.0, 1500.0),
            -120.0,
            [1000.0, 100.0, -1000.0],
            [0.0, 500.0, 2100.0],
            [1299.03810568, 1316.02540378, 480.38475773],
            [750.0, -279.42286341, -2032.05080757],
        ),
    ],
)
def test_transform_positions_to_local_field_coordinates(
    coordsys_origin,
    coordsys_rotation,
    utmx,
    utmy,
    expected_x,
    expected_y,
):
    xpos = np.array(utmx)
    ypos = np.array(utmy)
    transf_x, transf_y = transform_positions_to_local_field_coordinates(
        coordsys_origin, coordsys_rotation, xpos, ypos
    )
    assert_allclose(transf_x, expected_x)
    assert_allclose(transf_y, expected_y)


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
            np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
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
            np.array([[0.0, 9.0], [1.0, 9.0]], dtype=np.float32),
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
        assert np.array_equal(result, result_loc)


def test_that_calc_rho_for_2d_grid_layer_ignores_obs_outside_the_grid():
    """Test that observations positioned outside the grid with small ranges
    result in zero RHO values for all grid cells
    """
    nx, ny = 3, 3
    xinc, yinc = 1.0, 1.0

    obs_xpos = np.array([1.5, 100.0])  # Second obs is far outside.
    obs_ypos = np.array([1.5, 100.0])

    obs_main_range = np.array([1.0, 1.0])
    obs_perp_range = np.array([1.0, 1.0])
    obs_anisotropy_angle = np.array([0.0, 0.0])

    rho = calc_rho_for_2d_grid_layer(
        nx=nx,
        ny=ny,
        xinc=xinc,
        yinc=yinc,
        obs_xpos=obs_xpos,
        obs_ypos=obs_ypos,
        obs_main_range=obs_main_range,
        obs_perp_range=obs_perp_range,
        obs_anisotropy_angle=obs_anisotropy_angle,
    )

    assert rho.shape == (3, 3, 2)

    rho_inside = rho[:, :, 0]
    assert np.any(rho_inside > 0), "Observation inside the grid should have nonzero rho"

    rho_outside = rho[:, :, 1]
    assert np.all(rho_outside == 0.0), (
        "Observation outside the grid should have all zero rho values"
    )


def test_that_calc_rho_for_2d_grid_layer_validates_observation_lengths():
    with pytest.raises(
        ValueError,
        match="Number of coordinates must match number of observations",
    ):
        calc_rho_for_2d_grid_layer(
            nx=3,
            ny=3,
            xinc=1.0,
            yinc=1.0,
            obs_xpos=np.array([1.5]),
            obs_ypos=np.array([1.5, 2.5]),
            obs_main_range=np.array([1.0]),
            obs_perp_range=np.array([1.0]),
            obs_anisotropy_angle=np.array([0.0]),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"nx": 0}, "`nx` must be positive"),
        ({"ny": -1}, "`ny` must be positive"),
        ({"xinc": 0.0}, "`xinc` must be positive"),
        ({"yinc": -0.5}, "`yinc` must be positive"),
        (
            {"obs_main_range": np.array([0.0])},
            "All main-range values for all observations must be positive",
        ),
        (
            {"obs_perp_range": np.array([-1.0])},
            "All perpendicular-range values for all observations must be positive",
        ),
    ],
)
def test_that_calc_rho_for_2d_grid_layer_validates_inputs(kwargs, message):
    args = {
        "nx": 3,
        "ny": 3,
        "xinc": 1.0,
        "yinc": 1.0,
        "obs_xpos": np.array([1.5]),
        "obs_ypos": np.array([1.5]),
        "obs_main_range": np.array([1.0]),
        "obs_perp_range": np.array([1.0]),
        "obs_anisotropy_angle": np.array([0.0]),
    }
    args.update(kwargs)

    with pytest.raises(ValueError, match=message):
        calc_rho_for_2d_grid_layer(**args)
