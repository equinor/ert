from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import numpy as np
import resfo

from .field_file_format import ROFF_FORMATS, FieldFileFormat
from .grdecl_io import export_grdecl, import_bgrdecl, import_grdecl
from .roff_io import export_roff, import_roff

if TYPE_CHECKING:
    import numpy.typing as npt
    import xtgeo

_PathLike: TypeAlias = str | os.PathLike[str]


class AxisOrientation(Enum):
    """
    Defines the orientation of the grid axes for the used coordinate system,
    which can be either left-handed or right-handed.
    """

    LEFT_HANDED = auto()
    RIGHT_HANDED = auto()


class Shape(NamedTuple):
    nx: int
    ny: int
    nz: int


def _validate_array(
    kw: str, filename: _PathLike, vals: npt.NDArray[Any] | resfo.MessType
) -> npt.NDArray[Any]:
    if isinstance(vals, resfo.MessType):
        raise ValueError(f"{kw.strip()} in {filename} has incorrect type MESS")
    return vals


def _make_shape(sequence: npt.NDArray[Any]) -> Shape:
    return Shape(*(int(val) for val in sequence))


def read_mask(
    grid_path: _PathLike,
) -> tuple[npt.NDArray[np.bool_], Shape]:
    actnum = None
    shape = None
    actnum_coords: list[tuple[int, int, int]] = []
    with open(grid_path, "rb") as f:
        for entry in resfo.lazy_read(f):
            if actnum is not None and shape is not None:
                break

            keyword = str(entry.read_keyword()).strip()
            if actnum is None:
                if keyword == "COORDS":
                    coord_array = _validate_array(
                        "COORDS", grid_path, entry.read_array()
                    )
                    if coord_array[4]:
                        actnum_coords.append(
                            (coord_array[0], coord_array[1], coord_array[2])
                        )
                if keyword == "ACTNUM":
                    actnum = _validate_array("ACTNUM", grid_path, entry.read_array())
            if shape is None:
                if keyword == "GRIDHEAD":
                    arr = _validate_array("GRIDHEAD", grid_path, entry.read_array())
                    shape = _make_shape(arr[1:4])
                elif keyword == "DIMENS":
                    arr = _validate_array("DIMENS", grid_path, entry.read_array())
                    shape = _make_shape(arr[0:3])

    # Could possibly read shape from actnum_coords if they were read.
    if shape is None:
        raise ValueError(f"Could not load shape from {grid_path}")

    if actnum is None:
        if actnum_coords and len(actnum_coords) != np.prod(shape):
            actnum = np.ones(shape, dtype=bool)
            for coord in actnum_coords:
                actnum[coord[0] - 1, coord[1] - 1, coord[2] - 1] = False
        else:
            actnum = np.zeros(shape, dtype=bool)
    else:
        actnum = np.ascontiguousarray(np.logical_not(actnum.reshape(shape, order="F")))

    return actnum, shape


def get_shape(
    grid_path: _PathLike,
) -> Shape | None:
    shape = None
    with open(grid_path, "rb") as f:
        for entry in resfo.lazy_read(f):
            keyword = str(entry.read_keyword()).strip()
            if keyword == "GRIDHEAD":
                arr = _validate_array("GRIDHEAD", grid_path, entry.read_array())
                shape = _make_shape(arr[1:4])
            elif keyword == "DIMENS":
                arr = _validate_array("DIMENS", grid_path, entry.read_array())
                shape = _make_shape(arr[0:3])

    return shape


@dataclass(frozen=True)
class GridGeometry:
    nx: int
    ny: int
    nz: int
    axis_orientation: AxisOrientation | None = None
    xlength: float | None = None
    ylength: float | None = None
    xinc: float | None = None
    yinc: float | None = None
    rotation_angle: float | None = None
    origin: tuple[float, float] | None = None


def calculate_grid_geometry(grid: xtgeo.Grid) -> GridGeometry:
    """Calculate grid geometry from an XTGeo grid.

    Extracts geometric parameters including dimensions, cell increments,
    rotation angle, and origin coordinates needed for the grid.

    Args:
        grid: XTGeo Grid3D object

    Returns:
        GridGeometry with grid dimensions, increments, rotation, and origin
    """

    (nx, ny, nz) = grid.dimensions
    corner_indices = []
    axis_orientation = (
        AxisOrientation.RIGHT_HANDED
        if grid.ijk_handedness == "right"
        else AxisOrientation.LEFT_HANDED
    )

    if axis_orientation == AxisOrientation.LEFT_HANDED:
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

    if axis_orientation == AxisOrientation.LEFT_HANDED:
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
        angle = 90.0 if deltay1 > 0 else -90.0
    elif deltax1 > 0:
        angle = math.atan(deltay1 / deltax1) * 180.0 / math.pi
    elif deltax1 < 0:
        angle = (math.atan(deltay1 / deltax1) + math.pi) * 180.0 / math.pi

    return GridGeometry(
        nx=nx,
        ny=ny,
        nz=nz,
        xlength=xlength,
        ylength=ylength,
        xinc=xinc,
        yinc=yinc,
        rotation_angle=angle,
        origin=(x0, y0),
        axis_orientation=axis_orientation,
    )


def read_field(
    field_path: _PathLike,
    field_name: str,
    shape: Shape,
) -> np.ma.MaskedArray[Any, np.dtype[np.float32]]:
    path = Path(field_path)
    file_extension = path.suffix[1:].upper()
    try:
        file_format = FieldFileFormat[file_extension]
    except KeyError as err:
        raise ValueError(
            f'Could not read {field_path}. Unrecognized suffix "{file_extension}"'
        ) from err

    values: npt.NDArray[np.float32] | np.ma.MaskedArray[Any, np.dtype[np.float32]]
    if file_format in ROFF_FORMATS:
        values = import_roff(field_path, field_name)
    elif file_format == FieldFileFormat.GRDECL:
        values = import_grdecl(field_path, field_name, shape, dtype=np.float32)
    elif file_format == FieldFileFormat.BGRDECL:
        values = import_bgrdecl(field_path, field_name, shape)
    else:
        ext = path.suffix
        raise ValueError(f'Could not read {field_path}. Unrecognized suffix "{ext}"')

    return np.ma.MaskedArray(data=values, fill_value=np.nan)


def save_field(
    field: np.ma.MaskedArray[Any, np.dtype[np.float32]],
    field_name: str,
    output_path: _PathLike,
    file_format: FieldFileFormat,
) -> None:
    path = Path(output_path)
    os.makedirs(path.parent, exist_ok=True)
    if file_format in ROFF_FORMATS:
        export_roff(
            field,
            output_path,
            field_name,
            binary=file_format != FieldFileFormat.ROFF_ASCII,
        )
    elif file_format == FieldFileFormat.GRDECL:
        export_grdecl(field, output_path, field_name, binary=False)
    elif file_format == FieldFileFormat.BGRDECL:
        export_grdecl(field, output_path, field_name, binary=True)
    else:
        raise ValueError(f"Cannot export, invalid file format: {file_format}")


def transform_positions_to_local_field_coordinates(
    coordsys_origin: tuple[float, float],
    coordsys_rotation_angle: float,
    utmx: npt.NDArray[np.float64],
    utmy: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculates coordinate transformation from global to local coordinates.

    Args:
        coordys_origin: (x,y) coordinate of local coordinate
        origin in global coordinates.
        coordsys_rotation_angle: Angle for how much the local x-axis is rotated
        anti-clockwise relative to the global x-axis in degrees.
        utmx: vector of x-coordinates in global coordinates.
        utmy: vector of y-coordinates in global coordinates.

    Returns:
        First vector is local x-coordinates and second vector is local y-coordinates.
    """
    # Translate
    x1 = utmx - coordsys_origin[0]
    y1 = utmy - coordsys_origin[1]
    # Rotate
    # Input angle is the local coordinate systems rotation
    # anticlockwise relative to global x-axis in degrees
    rotation_of_grid = coordsys_rotation_angle
    rotation_angle = np.deg2rad(rotation_of_grid)
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    x2 = x1 * cos_theta + y1 * sin_theta
    y2 = -x1 * sin_theta + y1 * cos_theta
    return x2, y2


def transform_local_ellipse_angle_to_local_coords(
    coordsys_rotation_angle: float,
    ellipse_anisotropy_angle: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate angles relative to local coordinate system.

    Args:
        coordsys_rotation_angle: Local coordinate systems rotation angle
        relative to the global coordinate system.
        ellipse_anisotropy_angle: Vector of input angles in global coordinates.

    Returns:
        Vector of output angles relative to the local coordinate system.
    """
    # Both angles measured anti-clock from global coordinate systems x-axis in degrees
    return ellipse_anisotropy_angle - coordsys_rotation_angle


def localization_scaling_function(
    distances: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate scaling factor to be used as values in
    RHO matrix in distance-based localization.
    The scaling function implements the commonly
    used function published by Gaspari and Cohn.
    For input normalized distance >= 2, the value will be 0.

    Args:
        distances: Vector of values for normalized distances.

    Returns:
        Values of scaling factors for each value of input distance.
    """
    # "gaspari-cohn"
    # Commonly used in distance-based localization
    # Is exact 0 for normalized distance > 2.
    scaling_factor = distances
    d2 = distances**2
    d3 = d2 * distances
    d4 = d3 * distances
    d5 = d4 * distances
    s = -1 / 4 * d5 + 1 / 2 * d4 + 5 / 8 * d3 - 5 / 3 * d2 + 1
    scaling_factor[distances <= 1] = s[distances <= 1]
    s = (
        1 / 12 * d5
        - 1 / 2 * d4
        + 5 / 8 * d3
        + 5 / 3 * d2
        - 5 * distances
        + 4
        - 2 / 3 * 1 / distances
    )
    scaling_factor[(distances > 1) & (distances <= 2)] = s[
        (distances > 1) & (distances <= 2)
    ]
    scaling_factor[distances > 2] = 0.0

    return scaling_factor
