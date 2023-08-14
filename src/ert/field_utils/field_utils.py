from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional, Tuple, Union

import ecl_data_io
import numpy as np

from .grdecl_io import export_grdecl, import_bgrdecl, import_grdecl
from .roff_io import export_roff, import_roff

if TYPE_CHECKING:
    import numpy.typing as npt

_PathLike = Union[str, "os.PathLike[str]"]


class Shape(NamedTuple):
    nx: int
    ny: int
    nz: int


def get_mask(
    grid_path: Optional[_PathLike],
    shape: Optional[Shape] = None,
) -> Tuple[npt.NDArray[np.bool_], Shape]:
    if grid_path is not None:
        mask, shape = read_mask(grid_path, shape)
        if mask is None:
            return np.zeros(shape, dtype=bool), shape
        else:
            return mask, shape
    elif shape is not None:
        return np.zeros(shape, dtype=bool), shape

    raise ValueError("Could not load mask with no grid file or shape specified")


# pylint: disable=R0912
def read_mask(
    grid_path: _PathLike,
    shape: Optional[Shape] = None,
) -> Tuple[Optional[npt.NDArray[np.bool_]], Shape]:
    actnum = None
    actnum_coords: List[Tuple[int, int, int]] = []
    with open(grid_path, "rb") as f:
        for entry in ecl_data_io.lazy_read(f):
            if actnum is not None and shape is not None:
                break

            keyword = str(entry.read_keyword()).strip()
            if actnum is None:
                if keyword == "COORDS":
                    coord_array = entry.read_array()
                    if coord_array[4]:
                        actnum_coords.append(
                            (coord_array[0], coord_array[1], coord_array[2])
                        )
                if keyword == "ACTNUM":
                    actnum = entry.read_array()
            if shape is None:
                if keyword == "GRIDHEAD":
                    arr = entry.read_array()
                    shape = Shape(*(int(val) for val in arr[1:4]))
                elif keyword == "DIMENS":
                    arr = entry.read_array()
                    shape = Shape(*(int(val) for val in arr[0:3]))

    # Could possibly read shape from actnum_coords if they were read.
    if shape is None:
        raise ValueError(f"Could not load shape from {grid_path}")

    if actnum is None:
        if actnum_coords and len(actnum_coords) != np.prod(shape):
            actnum = np.ones(shape, dtype=bool)
            for coord in actnum_coords:
                actnum[coord[0] - 1, coord[1] - 1, coord[2] - 1] = False
    else:
        actnum = np.ascontiguousarray(np.logical_not(actnum.reshape(shape, order="F")))

    return actnum, shape


def get_shape(
    grid_path: _PathLike,
) -> Optional[Shape]:
    shape = None
    with open(grid_path, "rb") as f:
        for entry in ecl_data_io.lazy_read(f):
            keyword = str(entry.read_keyword()).strip()
            if keyword == "GRIDHEAD":
                arr = entry.read_array()
                shape = Shape(*(int(val) for val in arr[1:4]))
            elif keyword == "DIMENS":
                arr = entry.read_array()
                shape = Shape(*(int(val) for val in arr[0:3]))

    return shape


def read_field(
    field_path: _PathLike,
    field_name: str,
    mask: npt.NDArray[np.bool_],
    shape: Shape,
) -> np.ma.MaskedArray[Any, np.dtype[np.double]]:
    path = Path(field_path)
    ext = path.suffix
    values: Union[npt.NDArray[np.double], np.ma.MaskedArray[Any, np.dtype[np.double]]]
    if ext == ".roff":
        values = import_roff(field_path, field_name)
    elif ext == ".grdecl":
        values = import_grdecl(path, field_name, shape, dtype=np.double)
    elif ext == ".bgrdecl":
        values = import_bgrdecl(field_path, field_name, shape)
    else:
        raise ValueError(f'Could not read {field_path}. Unrecognized suffix "{ext}"')

    return np.ma.MaskedArray(data=values, mask=mask, fill_value=np.nan)  # type: ignore


def save_field(
    field: np.ma.MaskedArray[Any, np.dtype[np.double]],
    field_name: str,
    output_path: _PathLike,
    file_format: str,
) -> None:
    path = Path(output_path)
    os.makedirs(path.parent, exist_ok=True)
    if file_format in ["roff_binary", "roff_ascii", "roff"]:
        export_roff(field, output_path, field_name, binary=file_format != "roff_ascii")
    elif file_format == "grdecl":
        export_grdecl(field, output_path, field_name, binary=False)
    elif file_format == "bgrdecl":
        export_grdecl(field, output_path, field_name, binary=True)
    else:
        raise ValueError(f"Cannot export, invalid file format: {file_format}")
