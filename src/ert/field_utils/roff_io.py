from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, BinaryIO, TextIO

import numpy as np
import roffio  # type: ignore

if TYPE_CHECKING:
    from os import PathLike

    _PathLike = str | PathLike[str]


RMS_UNDEFINED_FLOAT = np.float32(-999.0)


def export_roff(
    data: np.ma.MaskedArray[Any, np.dtype[np.float32]],
    filelike: TextIO | BinaryIO | _PathLike,
    parameter_name: str,
    binary: bool,
) -> None:
    dimensions = data.shape
    data = np.flip(data, -1).ravel()  # type: ignore
    data = data.astype(np.float32).filled(RMS_UNDEFINED_FLOAT)  # type: ignore
    if not np.isfinite(data).all():
        raise ValueError(
            f"export of field {parameter_name!r} to {filelike}"
            " contained infinity or nan values"
        )

    file_contents = OrderedDict(
        {
            "filedata": {"filetype": "parameter"},
            "dimensions": {
                "nX": dimensions[0],
                "nY": dimensions[1],
                "nZ": dimensions[2],
            },
            "parameter": {"name": parameter_name, "data": data},
        }
    )
    roff_format = roffio.Format.BINARY if binary else roffio.Format.ASCII
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"casting array")
        roffio.write(filelike, file_contents, roff_format=roff_format)


def import_roff(
    filelike: TextIO | BinaryIO | _PathLike, name: str | None = None
) -> np.ma.MaskedArray[Any, np.dtype[np.float32]]:
    looking_for = {
        "dimensions": {
            "nX": None,
            "nY": None,
            "nZ": None,
        },
        "parameter": {
            "name": None,
            "data": None,
        },
    }

    def reset_parameter() -> None:
        looking_for["parameter"] = {"name": None, "data": None}

    def all_set() -> bool:
        return all(val is not None for v in looking_for.values() for val in v.values())

    def should_skip_parameter(key: tuple[str, str]) -> bool:
        return key[0] == "name" and name is not None and key[1] != name

    with roffio.lazy_read(filelike) as tag_generator:
        for tag, keys in tag_generator:
            if all_set():
                # We have already found the right parameter
                break
            if tag in looking_for:
                for key in keys:
                    if should_skip_parameter(key):
                        # Found a parameter, but not the one we are looking for
                        # reset and look on
                        reset_parameter()
                        break
                    if key[0] in looking_for[tag]:
                        looking_for[tag][key[0]] = key[1]

    data = looking_for["parameter"]["data"]
    if data is None:
        raise ValueError(f"Could not find roff parameter {name!r} in {filelike}")
    if not all_set():
        raise ValueError(
            f"Could not find dimensions for roff parameter {name!r} in {filelike}"
        )

    if isinstance(data, bytes) or np.issubdtype(data.dtype, np.uint8):
        raise ValueError("Ert does not support discrete roff field parameters")
    if np.issubdtype(data.dtype, np.integer):
        raise ValueError("Ert does not support discrete roff field parameters")
    if np.issubdtype(data.dtype, np.floating):
        if data.dtype == np.float64:
            # RMS can only work with 32 bit roff files
            data = data.astype(np.float32)
        dim = looking_for["dimensions"]
        if dim["nX"] * dim["nY"] * dim["nZ"] != data.size:
            raise ValueError(
                f"Field parameter {name!r} does not have correct number of"
                f" elements for given dimensions {dim} in {filelike}"
            )

        data = np.flip(data.reshape((dim["nX"], dim["nY"], dim["nZ"])), -1)
        return np.ma.masked_values(data, RMS_UNDEFINED_FLOAT)
    raise ValueError(
        f"Unexpected type of roff parameter {name} in {filelike}: {type(data)}"
    )
