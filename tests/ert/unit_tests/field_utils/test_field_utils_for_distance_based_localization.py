import numpy as np
import pytest
from numpy.testing import assert_allclose

from ert.field_utils import (
    localization_scaling_function,
    transform_local_ellipse_angle_to_local_coords,
    transform_positions_to_local_field_coordinates,
)


@pytest.mark.parametrize(
    "nvalues",
    [
        10,
        25,
    ],
)
def test_localization_scaling_function(snapshot, nvalues):
    distances = np.linspace(0, 2.5, num=nvalues, endpoint=True, dtype=np.float64)
    scaling_values = localization_scaling_function(distances)
    snapshot.assert_match(str(scaling_values) + "\n", "testdata_scaling_values.txt")


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
