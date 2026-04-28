from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import numpy as np
import pandas as pd
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
    Defines the grid index origin. It is ONLY related to the
    order of the grid index, not the coordinates axis.
    For a non-rotated grid, left-handed means that grid index origin
    is lower left corner, the same as the local coordinate origin.
    I-index is increasing in direction EAST or to the right,
    J-index is increasing in direction NORTH or upwards
    while K-index is increasing with depth (or into the paper/screen).
    For C-indexing, the flatten index for the 3D field parameter
    with cell index (I,J,K) is index = K + J*NZ + I*NZ*NY.
    For right-handed, grid index origin is upper left corner,
    the I-index is increasing in direction EAST or to the right,
    the J-index is increasing in direction SOUTH or from top to bottom
    of the paper/screen. For C-indexing, the flatten index for the
    3D field parameter with cell index (I,J,K)
    is index = K + (NY-J-1)*NZ + I*NZ*NY.
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


def get_shape(
    grid_path: _PathLike,
) -> Shape | None:
    shape = None
    with Path(grid_path).open("rb") as f:
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
class ErtboxParameters:
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


def calculate_ertbox_parameters(grid: xtgeo.Grid) -> ErtboxParameters:
    """Calculate ERTBOX grid parameters from an XTGeo grid.

    Extracts geometric parameters including dimensions, cell increments,
    rotation angle, and origin coordinates needed for ERTBOX. Get the grid
    index origin, called AxisOrientation which define the direction of the
    J-index. For right-handed, the J-index increases in direction SOUTH
    for non-rotated grid, for lef-handed, the J-index increases in
    direction NORTH. Note that AxisOrientation does not change the
    local coordinate axis which has its origin at lower left corner
    with x-axis in direction EAST and y-axis in direction NORTH for
    non-rotated ERTBOX grid.

    Args:
        grid: XTGeo Grid3D object

    Returns:
        ErtboxParameters with grid dimensions, increments, rotation, origi
        and grid index origin.
    """

    (nx, ny, nz) = grid.dimensions
    corner_indices = []
    axis_orientation = (
        AxisOrientation.RIGHT_HANDED
        if grid.ijk_handedness == "right"
        else AxisOrientation.LEFT_HANDED
    )

    if axis_orientation == AxisOrientation.LEFT_HANDED:
        # Grid index origin is at lower left corner
        # ERTBOX local coordinate origin is lower left corner
        origin_cell = (1, 1, 1)
        x_direction_cell = (nx, 1, 1)
        y_direction_cell = (1, ny, 1)
    else:
        # Grid index origin is at upper left corner
        # ERTBOX local coordinate origin is lower left corner
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
    utmx: npt.NDArray[np.floating],
    utmy: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
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
    rotation_of_ertbox = coordsys_rotation_angle
    rotation_angle = np.deg2rad(rotation_of_ertbox)
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


def gaspari_cohn(
    distances: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Gaspari--Cohn distance-based localization scaling function.

    For each normalised distance d, returns a scaling factor in [0, 1]
    used as elements in the localization matrix (rho).
    For d >= 2 the value is 0.

    This is an implementation of Eq. (4.10) in Section 4.3
    ("Compactly supported 5th-order piecewise rational functions") of
    Gaspari and Cohn (1999).

    References
    ----------
    Gaspari, G. and Cohn, S.E. (1999), Construction of correlation functions
    in two and three dimensions. Q.J.R. Meteorol. Soc., 125: 723-757.
    https://doi.org/10.1002/qj.49712555417

    Parameters
    ----------
    distances : np.ndarray
        Vector of values for normalized distances.

    Returns
    -------
    np.ndarray
        Values of scaling factors for each value of input distance.

    Examples
    --------
    The function equals 1 at d=0, 5/24 at d=1, and 0 for d>=2:

    >>> import numpy as np
    >>> gaspari_cohn(np.array([0.0, 1.0, 2.0, 3.0]))
    array([1.        , 0.20833333, 0.        , 0.        ])

    The input array is not modified:

    >>> d = np.array([0.5, 1.5])
    >>> _ = gaspari_cohn(d)
    >>> d
    array([0.5, 1.5])
    """
    if not np.all(distances >= 0):
        raise ValueError(f"Distances must be positive. Min: {np.min(distances)}")
    scaling_factor = np.zeros_like(distances)

    d2 = distances**2
    d3 = d2 * distances
    d4 = d3 * distances
    d5 = d4 * distances

    near = distances <= 1
    scaling_factor[near] = (
        -1 / 4 * d5[near] + 1 / 2 * d4[near] + 5 / 8 * d3[near] - 5 / 3 * d2[near] + 1
    )

    mid = (distances > 1) & (distances <= 2)
    scaling_factor[mid] = (
        1 / 12 * d5[mid]
        - 1 / 2 * d4[mid]
        + 5 / 8 * d3[mid]
        + 5 / 3 * d2[mid]
        - 5 * distances[mid]
        + 4
        - 2 / 3 / distances[mid]
    )

    # Clip to [0, 1] to suppress tiny negative artefacts from
    # floating-point arithmetic at the d=2 boundary.
    np.clip(scaling_factor, 0.0, 1.0, out=scaling_factor)
    return scaling_factor


def calc_rho_for_2d_grid_layer(
    *,
    nx: int,
    ny: int,
    xinc: float,
    yinc: float,
    obs_xpos: npt.NDArray[np.floating],
    obs_ypos: npt.NDArray[np.floating],
    obs_main_range: npt.NDArray[np.floating],
    obs_perp_range: npt.NDArray[np.floating],
    obs_anisotropy_angle: npt.NDArray[np.floating],
    axis_orientation: AxisOrientation = AxisOrientation.RIGHT_HANDED,
) -> npt.NDArray[np.floating]:
    """Calculate elements of the localization matrix (rho) for a 2D grid layer.

    For each observation, the distance to every grid cell centre is computed
    and passed through the Gaspari--Cohn scaling function to obtain rho.
    Only lateral distances (horizontal distances in the (x, y) plane,
    ignoring depth) are considered, so every depth layer of a 3D grid
    shares the same cell centres and produces identical rho values;
    a single 2D calculation therefore covers all depth layers.
    All observation positions are given in the local grid coordinate
    system.

    Each observation n is described by its position
    (obs_xpos[n], obs_ypos[n]) and its localization ellipse
    (obs_main_range[n], obs_perp_range[n], obs_anisotropy_angle[n]).

    Grid cells are addressed by a flat index m that encodes the 2D cell
    index (i, j):
        m = j + i * ny                (left-handed grid indexing)
        m = (ny - j - 1) + i * ny    (right-handed grid indexing)

    The 2D distance from observation n to grid cell m = (i, j) is:
        d[m, n] = dist(
            (obs_xpos[n], obs_ypos[n]),
            ((i + 0.5) * xinc, (j + 0.5) * yinc),
        )

    where (i + 0.5) * xinc and (j + 0.5) * yinc are the x- and y-coordinates
    of the centre of grid cell (i, j) in the local coordinate system.

    The localization matrix element for cell m and observation n is:
        rho[m, n] = gaspari_cohn(d[m, n])

    Parameters
    ----------
    nx : int
        Number of grid cells in x-direction of local coordinate system.
    ny : int
        Number of grid cells in y-direction of local coordinate system.
    xinc : float
        Grid cell size in x-direction.
    yinc : float
        Grid cell size in y-direction.
    obs_xpos : np.ndarray
        Observations x coordinates in local coordinates.
    obs_ypos : np.ndarray
        Observations y coordinates in local coordinates.
    obs_main_range : np.ndarray
        Semi-axis length of the localization ellipse along the principal axis
        (the axis oriented at ``obs_anisotropy_angle`` relative to the
        local x-axis).
    obs_perp_range : np.ndarray
        Semi-axis length of the localization ellipse perpendicular to the
        principal axis. Equal to ``obs_main_range`` gives a circle; smaller
        gives an elongated ellipse.
    obs_anisotropy_angle : np.ndarray
        Orientation of the principal axis of the localization ellipse in
        degrees relative to the local x-axis. An angle of 0 aligns the
        principal axis with the x-axis of the local coordinate system.
    axis_orientation : AxisOrientation, optional
        Grid index origin convention. Default is ``RIGHT_HANDED``.

    Returns
    -------
    np.ndarray
        Localization matrix (rho) of shape ``(nx, ny, nobs)`` for one
        layer of a 3D grid or for a 2D surface grid.
    """
    if nx <= 0:
        raise ValueError("`nx` must be positive")
    if ny <= 0:
        raise ValueError("`ny` must be positive")

    if xinc <= 0.0:
        raise ValueError("`xinc` must be positive")
    if yinc <= 0.0:
        raise ValueError("`yinc` must be positive")

    if axis_orientation == AxisOrientation.RIGHT_HANDED:
        # y coordinate decreases from max to min
        y_local = (np.arange(ny - 1, -1, -1) + 0.5) * yinc
    else:
        # y coordinate increases from min to max
        y_local = (np.arange(ny) + 0.5) * yinc

    # Validate that all observation arrays are 1-D
    for name, arr in [
        ("obs_xpos", obs_xpos),
        ("obs_ypos", obs_ypos),
        ("obs_main_range", obs_main_range),
        ("obs_perp_range", obs_perp_range),
        ("obs_anisotropy_angle", obs_anisotropy_angle),
    ]:
        if arr.ndim != 1:
            raise ValueError(f"`{name}` must be 1-D, got {arr.ndim}-D")

    nobs = obs_xpos.shape[0]
    if obs_ypos.shape[0] != nobs:
        raise ValueError("Number of coordinates must match number of observations")
    if obs_anisotropy_angle.shape[0] != nobs:
        raise ValueError(
            "Number of ellipse orientation angles must match number of observations"
        )
    if obs_main_range.shape[0] != nobs:
        raise ValueError(
            "Number of ellipse main range values must match number of observations"
        )
    if obs_perp_range.shape[0] != nobs:
        raise ValueError(
            "Number of ellipse perpendicular range values must match number"
            " of observations"
        )
    if np.any(obs_main_range <= 0.0):
        raise ValueError("All main-range values for all observations must be positive")
    if np.any(obs_perp_range <= 0.0):
        raise ValueError(
            "All perpendicular-range values for all observations must be positive"
        )

    # Observation coordinates and parameters
    obs_xpos = obs_xpos[np.newaxis, np.newaxis, :]  # (1, 1, nobs)
    obs_ypos = obs_ypos[np.newaxis, np.newaxis, :]  # (1, 1, nobs)
    obs_main_range = obs_main_range[np.newaxis, np.newaxis, :]  # (1, 1, nobs)
    obs_perp_range = obs_perp_range[np.newaxis, np.newaxis, :]  # (1, 1, nobs)
    # (1, 1, nobs)
    obs_anisotropy_angle = obs_anisotropy_angle[np.newaxis, np.newaxis, :]

    # Center points of each grid cell in field parameter grid
    x_local = (np.arange(nx) + 0.5) * xinc

    # Use 3D broadcasting to avoid allocating memory
    x_local = x_local[:, np.newaxis, np.newaxis]  # (nx, 1, 1)
    y_local = y_local[np.newaxis, :, np.newaxis]  # (1, ny, 1)

    # Compute displacement in x and y directions between each grid point
    # and each observation point.
    dX = x_local - obs_xpos  # (nx, 1, nobs)
    dY = y_local - obs_ypos  # (1, ny, nobs)

    # Compute rotation parameters
    rotation = np.deg2rad(obs_anisotropy_angle)  # (1, 1, nobs)
    cos_angle = np.cos(rotation)  # (1, 1, nobs)
    sin_angle = np.sin(rotation)  # (1, 1, nobs)

    # Rotate displacements into a coordinate system aligned with the semi-axes
    # of the influence ellipse for each observation. The new x-axis aligns
    # with the anisotropy angle, and the new y-axis is perpendicular to it.
    # The rotated displacements are then scaled by the respective semi-axis
    # lengths (ranges) so that points on the ellipse boundary have a distance of 1.
    dX_ellipse = (dX * cos_angle + dY * sin_angle) / obs_main_range
    dY_ellipse = (-dX * sin_angle + dY * cos_angle) / obs_perp_range

    # Compute distances in the elliptical coordinate system
    distances = np.hypot(dX_ellipse, dY_ellipse)  # (nx, ny, nobs)

    return gaspari_cohn(distances)


def transform_observation_locations(
    obs_loc_df: pd.DataFrame, ertbox_params: ErtboxParameters
) -> npt.NDArray[np.float32] | None:
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
            return np.column_stack((xpos[inside_box], ypos[inside_box])).astype(
                np.float32
            )
    return None
