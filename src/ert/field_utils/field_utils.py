from __future__ import annotations

import math
import os
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
    import xtgeo  # type: ignore

_PathLike: TypeAlias = str | os.PathLike[str]


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
