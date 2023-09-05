from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, BinaryIO, TextIO, Union

import numpy as np
import roffio  # type: ignore

if TYPE_CHECKING:
    from os import PathLike

_PathLike = Union[str, "PathLike[str]"]


def export_roff(
    data: np.ma.MaskedArray[Any, np.dtype[np.float32]],
    filelike: Union[TextIO, BinaryIO, _PathLike],
    parameter_name: str,
    binary: bool = True,
) -> None:
    dimensions = data.shape
    data = np.flip(data, -1).ravel()  # type: ignore
    data = data.filled(np.nan)  # type: ignore

    data_ = OrderedDict(
        {
            "filedata": {"filetype": "parameter"},
            "dimensions": {
                "nX": dimensions[0],
                "nY": dimensions[1],
                "nZ": dimensions[2],
            },
            "parameter": {"name": parameter_name},
        }
    )
    data_["parameter"]["data"] = data
    roff_format = roffio.Format.BINARY if binary else roffio.Format.ASCII
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"casting array")
        roffio.write(filelike, data_, roff_format=roff_format)
