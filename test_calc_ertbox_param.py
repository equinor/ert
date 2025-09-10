import math
from dataclasses import dataclass

import pytest
import xtgeo


@dataclass(frozen=True)
class ErtboxParameters:
    nx: int
    ny: int
    nz: int
    xlength: float
    ylength: float
    xinc: float
    yinc: float
    rotation_angle: float
    origin: tuple[float, float]


def calculate_ertbox_parameters(
    grid: xtgeo.Grid, left_handed: bool = False
) -> ErtboxParameters:
    """Calculate ERTBOX grid parameters from an XTGeo grid.

    Extracts geometric parameters including dimensions, cell increments,
    rotation angle, and origin coordinates needed for ERTBOX.

    Args:
        grid: XTGeo Grid3D object
        left_handed: If True, use left-handed coordinate system (default: False)

    Returns:
        ErtboxParameters with grid dimensions, increments, rotation, and origin
    """

    (nx, ny, nz) = grid.dimensions

    corner_indices = []

    if left_handed:
        origin_cell = (1, 1, 1)
        x_direction_cell = (nx, 1, 1)
        y_direction_cell = (1, ny, 1)
    else:
        origin_cell = (1, ny, 1)
        x_direction_cell = (nx, ny, 1)
        y_direction_cell = (1, 1, 1)

    corner_indices = [origin_cell, x_direction_cell, y_direction_cell]

    # List with 3 elements, where each element contains the coordinates
    # for all 8 corners of a single grid cell.
    coord_cell = []

    for corner_index in corner_indices:
        # Get real-world (x,y,z) coordinates for all 8 corners of this grid cell
        # Returns 24 values: [x0,y0,z0, x1,y1,z1, ..., x7,y7,z7]
        coord = grid.get_xyz_cell_corners(ijk=corner_index, activeonly=False)
        coord_cell.append(coord)

    if left_handed:
        # Origin: cell (1,1,1), corner 0
        x0 = coord_cell[0][0]
        y0 = coord_cell[0][1]

        # X-direction: cell (nx,1,1), corner 1
        x1 = coord_cell[1][3]
        y1 = coord_cell[1][4]

        # Y-direction: cell (1,ny,1), corner 2
        x2 = coord_cell[2][6]
        y2 = coord_cell[2][7]
    else:
        # Origin: cell (1,ny,1), corner 2
        x0 = coord_cell[0][6]
        y0 = coord_cell[0][7]

        # X-direction: cell (nx,ny,1), corner 3
        x1 = coord_cell[1][9]
        y1 = coord_cell[1][10]

        # Y-direction: cell (1,1,1), corner 0
        x2 = coord_cell[2][0]
        y2 = coord_cell[2][1]

    deltax1 = x1 - x0
    deltay1 = y1 - y0

    deltax2 = x2 - x0
    deltay2 = y2 - y0

    xlength = math.sqrt(deltax1**2 + deltay1**2)
    ylength = math.sqrt(deltax2**2 + deltay2**2)
    xinc = xlength / nx
    yinc = ylength / ny

    if math.fabs(deltax1) < 0.00001:
        angle = 90 if deltay1 > 0 else -90
    elif deltax1 > 0:
        angle = math.atan(deltay1 / deltax1) * 180.0 / math.pi
    elif deltax1 < 0:
        angle = (math.atan(deltay1 / deltax1) + math.pi) * 180.0 / math.pi

    return ErtboxParameters(
        nx=nx,
        ny=ny,
        nz=nz,
        xlength=xlength,
        ylength=ylength,
        xinc=xinc,
        yinc=yinc,
        rotation_angle=angle,
        origin=(x0, y0),
    )


@pytest.mark.parametrize(
    "origin, increment, rotation, flip",
    [
        ((1000.0, 2000.0, 1000.0), (100.0, 150.0, 10.0), 90, 1),
        ((1000.0, 2000.0, 1000.0), (100.0, 150.0, 10.0), 45, -1),
        ((1000.0, 2000.0, 1000.0), (100.0, 150.0, 10.0), 180, 1),
        ((1000.0, 2000.0, 1000.0), (100.0, 150.0, 10.0), -30, 1),
        ((0, 0, -10.0), (1, 1, 1), 0, 1),
    ],
)
def test_calculate_ertbox_parameters_synthetic_grid(origin, increment, rotation, flip):
    """Test calculate_ertbox_parameters with a synthetic box grid."""

    # Create a synthetic box grid with rotation
    grid = xtgeo.create_box_grid(
        dimension=(5, 4, 3),  # nx, ny, nz
        origin=origin,  # x0, y0, z0
        oricenter=False,  # origin at corner, not center
        increment=increment,  # dx, dy, dz
        rotation=rotation,  # rotation in degrees
        flip=flip,  # 1 for right-handed, -1 for left-handed
    )
    params = calculate_ertbox_parameters(grid)

    # Test calculated increments (should match input within tolerance)
    tolerance = 1e-10
    expected_xinc = increment[0]
    expected_yinc = increment[1]
    assert abs(params.xinc - expected_xinc) < tolerance, (
        f"Expected xinc={expected_xinc}, got {params.xinc}"
    )
    assert abs(params.yinc - expected_yinc) < tolerance, (
        f"Expected yinc={expected_yinc}, got {params.yinc}"
    )

    # Test rotation angle (should match input within tolerance)
    expected_angle = rotation
    angle_tolerance = 1e-6  # degrees
    assert abs(params.rotation_angle - expected_angle) < angle_tolerance, (
        f"Expected rotation={expected_angle}°, got {params.rotation_angle:.6f}°"
    )

    # Test that xlength and ylength are consistent with grid dimensions and increments
    expected_xlength = params.nx * expected_xinc
    expected_ylength = params.ny * expected_yinc
    assert abs(params.xlength - expected_xlength) < tolerance, (
        f"Expected xlength={expected_xlength}, got {params.xlength}"
    )
    assert abs(params.ylength - expected_ylength) < tolerance, (
        f"Expected ylength={expected_ylength}, got {params.ylength}"
    )

    # Test that xinc and yinc are positive
    assert params.xinc > 0, f"xinc should be positive, got {params.xinc}"
    assert params.yinc > 0, f"yinc should be positive, got {params.yinc}"

    # Test that rotation angle is in reasonable range
    assert -180 <= params.rotation_angle <= 180, (
        f"Rotation angle should be between -180° and 180°, got {params.rotation_angle}"
    )

    # Test grid dimensions
    assert params.nx == 5
    assert params.ny == 4
    assert params.nz == 3
