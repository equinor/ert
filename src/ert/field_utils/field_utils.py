from __future__ import annotations

import math
import os
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias

import numpy as np
import resfo
from pydantic.dataclasses import dataclass

from .field_file_format import ROFF_FORMATS, FieldFileFormat
from .grdecl_io import export_grdecl, import_bgrdecl, import_grdecl
from .roff_io import export_roff, import_roff

if TYPE_CHECKING:
    import numpy.typing as npt
    import xtgeo

_PathLike: TypeAlias = str | os.PathLike[str]


class Shape(NamedTuple):
    nx: int
    ny: int
    nz: int


class ScalingFunctions(StrEnum):
    gaspari_cohn = "gaspari_cohn"
    gaussian = "gaussian"
    exponential = "exponential"


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
class ErtboxParameters:
    nx: int
    ny: int
    nz: int
    xlength: float | None = None
    ylength: float | None = None
    xinc: float | None = None
    yinc: float | None = None
    rotation_angle: float | None = None
    origin: tuple[float, float] | None = None


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

    return np.ma.MaskedArray(data=values, fill_value=np.nan)  # type: ignore


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
    """
    Calculates the coordinates of the input (x,y) positions
    in local coordinates defined by the local coordinate system.
    The input coordinates are assumed to be in global (utm) coordinates.
    The return values are coordinate values in local coordinate system.
    """
    # Translate
    x1 = utmx - coordsys_origin[0]
    y1 = utmy - coordsys_origin[1]
    # Rotate
    # Input angle is the local coordinate systems rotation
    # anticlockwise relative to global x-axis in degrees
    rotation_of_ertbox = coordsys_rotation_angle
    rotation_angle = rotation_of_ertbox * np.pi / 180.0
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    x2 = x1 * cos_theta + y1 * sin_theta
    y2 = -x1 * sin_theta + y1 * cos_theta
    return x2, y2


def transform_local_ellipse_angle_to_local_coords(
    coordsys_rotation_angle: float,
    ellipse_anisotropy_angle: npt.NDArray[np.double],
) -> npt.NDArray[np.double]:
    """
    The input angles for orientation of the localization ellipses
    are relative to the global (utm) coordinates.
    The output is the angles relative to the local coordinate system.
    """
    # Both angles measured anti-clock from global coordinate systems x-axis in degrees
    ellipse_anisotropy_transformed = ellipse_anisotropy_angle - coordsys_rotation_angle
    return ellipse_anisotropy_transformed


def localization_scaling_function(
    distances: npt.NDArray[np.float64],
    scaling_func: ScalingFunctions = ScalingFunctions.gaspari_cohn,
) -> npt.NDArray[np.float64]:
    """
    Description: Calculate scaling factor to be used as values in
    RHO matrix in distance-based localization.
    Take as input array with normalized distances
    and returns scaling factors.

    :param distances: Array with distances
    :type distances: npt.NDArray[np.float64]
    :param scaling_func: Name of the scaling function type
    :type scaling_func: ScalingFunctions
    :return: Array with scaling values
    :rtype: NDArray[float64]
    """
    assert isinstance(scaling_func, ScalingFunctions)
    if scaling_func == ScalingFunctions.gaussian:
        # Same function as often used for gaussian variograms
        # Never exact 0. Maybe have a cutoff for normalized distance
        # equal to 2.5?
        scaling_factor = np.exp(-3.0 * distances**2)
    elif scaling_func == ScalingFunctions.exponential:
        # Same function as often used for exponential variograms
        # Never exact 0. Maybe have a cutoff for normalized distance
        # equal to 2.5?
        scaling_factor = np.exp(-3.0 * distances)
    else:
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


def calc_rho_for_2d_grid_layer(
    nx: int,
    ny: int,
    xinc: float,
    yinc: float,
    obs_xpos: npt.NDArray[np.double],
    obs_ypos: npt.NDArray[np.double],
    obs_main_range: npt.NDArray[np.double],
    obs_perp_range: npt.NDArray[np.double],
    obs_anisotropy_angle: npt.NDArray[np.double],
    scaling_function: ScalingFunctions = ScalingFunctions.gaspari_cohn,
) -> npt.NDArray[np.double]:
    """
    Description:
    Calculate scaling values (RHO matrix elements) for a set of observations
    with associated localization ellipse. The method will first
    calculate the distances from each observation position to each grid cell
    center point of all grid cells for a 2D grid.
    The localization method will only consider lateral distances, and it is
    therefore sufficient to calculate the distances in 2D.
    All input observation positions are in the local grid coordinate system
    to simplify the calculation of the distances.
    The method loops over all observations and is not optimally
    implemented with numpy.

    The position: xpos[n], ypos[n] and
    localization ellipse defined by obs_main_range[n],obs_perp_range[n],
    obs_anisotropy_angle[n]) refers to observation[n].

    The distance between an observation with index n and a grid cell (i,j) is
    d[m,n] = dist((xpos_obs[n],ypos_obs[n]),(xpos_field[i,j],ypos_field[i,j]))

    RHO[[m,n] = scaling(d)
    where m = j + i * ny for left-handed grid index origo and
          m = (ny - j - 1) + i * ny for right-handed grid index origo
    Note that since d[m,n] does only depend on observation index n and
    grid cell index (i,j). The values for RHO is
    calculated for the combination ((i,j), n) and this covers
    one grid layer in ertbox grid or a 2D surface grid.

    :param nx: Number of grid cells in x-direction of local coordinate system.
    :type nx: int
    :param ny: Number of grid cells in y-direction of local coordinate system.
    :type ny: int
    :param xinc: Grid cell size in x-direction.
    :type xinc: float
    :param yinc: Grid cell size in y-direction.
    :type yinc: float
    :param obs_xpos: Observations x coordinates
    :type obs_xpos: npt.NDArray[np.double]
    :param obs_ypos: Observatiopns y coordinates
    :type obs_ypos: npt.NDArray[np.double]
    :param obs_main_range: Localization ellipse first range
    :type obs_main_range: npt.NDArray[np.double]
    :param obs_perp_range: Localization ellipse second range
    :type obs_perp_range: npt.NDArray[np.double]
    :param obs_anisotropy_angle: Localization ellipse orientation
    :type obs_anisotropy_angle: npt.NDArray[np.double]
    :param scaling_function: Name of scaling function
    :type scaling_function: ScalingFunctions
    :return: Rho matrix values for one layer of the 3D ertbox grid
    :rtype: NDArray[double]
    """
    # Center points of each grid cell in field parameter grid
    handedness = "right"  # Hard-coded to right-handed grid indexing
    x_local = (np.arange(nx, dtype=np.float64) + 0.5) * xinc
    if handedness == "right":
        # y coordinate descreases from max to min
        y_local = (np.arange(ny - 1, -1, -1, dtype=np.float64) + 0.5) * yinc
    else:
        # y coordinate increases from min to max
        y_local = (np.arange(ny, dtype=np.float64) + 0.5) * yinc
    mesh_x_coord, mesh_y_coord = np.meshgrid(x_local, y_local, indexing="ij")

    # Number of observations
    nobs = len(obs_xpos)
    assert nobs == len(obs_ypos)
    assert nobs == len(obs_anisotropy_angle)
    assert nobs == len(obs_main_range)
    assert nobs == len(obs_perp_range)

    # Expand grid coordinates to match observations
    mesh_x_coord_flat = mesh_x_coord.flatten()[:, np.newaxis]  # (nx * ny, 1)
    mesh_y_coord_flat = mesh_y_coord.flatten()[:, np.newaxis]  # (nx * ny, 1)

    # Observation coordinates and parameters
    obs_xpos = obs_xpos[np.newaxis, :]  # (1, nobs)
    obs_ypos = obs_ypos[np.newaxis, :]  # (1, nobs)
    obs_main_range = obs_main_range[np.newaxis, :]  # (1, nobs)
    obs_perp_range = obs_perp_range[np.newaxis, :]  # (1, nobs)
    obs_anisotropy_angle = obs_anisotropy_angle[np.newaxis, :]  # (1, nobs)

    # Compute displacement between grid points and observations
    dX = mesh_x_coord_flat - obs_xpos  # (nx * ny, nobs)
    dY = mesh_y_coord_flat - obs_ypos  # (nx * ny, nobs)

    # Compute rotation parameters
    rotation = obs_anisotropy_angle * np.pi / 180.0  # (1, nobs)
    cos_angle = np.cos(rotation)  # (1, nobs)
    sin_angle = np.sin(rotation)  # (1, nobs)

    # Rotate and scale displacements to local coordinate system defined
    # by the two half axes of the influence ellipse. First coordinate (local x) is in
    # direction defined by anisotropy angle and local y is perpendicular to that.
    # Scale the distance by the ranges to get a normalized distance
    # (with value 1 at the edge of the ellipse)
    dX_ellipse = (dX * cos_angle + dY * sin_angle) / obs_main_range  # (nx * ny, nobs)
    dY_ellipse = (-dX * sin_angle + dY * cos_angle) / obs_perp_range  # (nx * ny, nobs)

    # Compute distances in the elliptical coordinate system
    distances = np.sqrt(dX_ellipse**2 + dY_ellipse**2)  # (nx * ny, nobs)

    # Apply the scaling function
    rho_one_layer = localization_scaling_function(
        distances, scaling_func=scaling_function
    )  # (nx * ny, nobs)
    rho_2D = rho_one_layer.reshape((nx, ny, nobs))
    return rho_2D
