from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, TextIO, Tuple, Union

import ecl_data_io
import numpy as np
import numpy.typing as npt


def split_line_no_string(line: str) -> Iterator[str]:
    for w in line.split():
        if w.startswith("--"):
            return
        yield w


def until_space(string: str) -> str:
    """
    returns the given string until the first space.
    Similar to string.split(max_split=1)[0] except
    initial spaces are not ignored:
    >>> until_space(" hello")
    ''
    >>> until_space("hello world")
    'hello'

    """
    result = ""
    for w in string:
        if w.isspace():
            return result
        result += w
    return result


def interpret_token(val: str) -> list[str]:
    """
    Interpret a eclipse token, tries to interpret the
    value in the following order:
    * string literal
    * keyword
    * repreated keyword
    * number

    If the token cannot be matched, we default to returning
    the uninterpreted token.

    >>> interpret_token("3")
    ['3']
    >>> interpret_token("1.0")
    ['1.0']
    >>> interpret_token("'hello'")
    ['hello']
    >>> interpret_token("PORO")
    ['PORO']
    >>> interpret_token("3PORO")
    ['3PORO']
    >>> interpret_token("3*PORO")
    ['PORO', 'PORO', 'PORO']
    >>> interpret_token("3*'PORO '")
    ['PORO ', 'PORO ', 'PORO ']
    >>> interpret_token("3'PORO '")
    ["3'PORO '"]

    """
    if val[0] == "'" and val[-1] == "'":
        # A string literal
        return [val[1:-1]]
    if val[0].isalpha():
        # A keyword
        return [val]
    if "*" in val:
        multiplicand, value = val.split("*")
        return interpret_token(value) * int(multiplicand)
    return [val]


IGNORE_ALL = None


@contextmanager
def open_grdecl(
    grdecl_file: Union[str, os.PathLike[str]],
    keywords: list[str],
) -> Iterator[Iterator[Tuple[str, str]]]:
    """Generates tuples of keyword and values in records of a grdecl file.

    The format of the file must be that of the GRID section of a eclipse input
    DATA file.

    The records looked for must be "simple" ie.  start with the keyword, be
    followed by single word values and ended by a slash ('/').

    .. code-block:: none

        KEYWORD
        value value value /

    reading the above file with :code:`open_grdecl("filename.grdecl",
    keywords="KEYWORD")` will generate :code:`[("KEYWORD", ["value", "value",
    "value"])]`

    open_grdecl does not follow includes, obey skips, parse MESSAGE commands or
    make exception for groups and subrecords.

    Raises:
        ValueError: when end of file is reached without terminating a keyword,
            or the file contains an unrecognized (or ignored) keyword.

    Args:
        grdecl_file (str): file path
        keywords (List[str]): Which keywords to look for, these are expected to
        be at the start of a line in the file  and the respective values
        following on subsequent lines separated by whitespace. Reading of a
        keyword is completed by a final '\'. See example above.
    """

    def read_grdecl(grdecl_stream: TextIO) -> Iterator[Tuple[str, str]]:
        words: List[str] = []
        keyword = None
        nonlocal keywords
        keywords = [until_space(keyword) for keyword in keywords]

        line_no = 1
        line = grdecl_stream.readline()

        while line:
            if line is None:
                break

            if keyword is None:
                snubbed = line[0 : min(8, len(until_space(line)))]
                simple_matched_keywords = [kw for kw in keywords if kw == snubbed]
                if simple_matched_keywords:
                    keyword = simple_matched_keywords[0]
            else:
                for word in split_line_no_string(line):
                    if word == "/":
                        yield (keyword, words)
                        keyword = None
                        words = []
                        break
                    words += interpret_token(word)
            line = grdecl_stream.readline()
            line_no += 1

        if keyword is not None:
            raise ValueError(f"Reached end of stream while reading {keyword}")

    with open(grdecl_file, "r", encoding="utf-8") as stream:
        yield read_grdecl(stream)


def read_grdecl_3d_property(
    filename: Union[str, os.PathLike[str]],
    keyword: str,
    dimensions: Tuple[int, int, int],
    dtype: npt.DTypeLike = float,
) -> npt.NDArray[np.float32]:
    """
    Read a 3d grid property from a grdecl file, see open_grdecl for description
    of format.

    Args:
        filename (pathlib.Path or str): File in grdecl format.
        keyword (str): The keyword of the property in the file
        dimensions ((int,int,int)): Triple of the size of grid.
        dtype (data-type, optional): The datatype to be read, ie., float.

    Raises:
        xtgeo.KeywordNotFoundError: If keyword is not found in the file.

    Returns:
        numpy array with given dimensions and data type read
        from the grdecl file.
    """
    result = None

    with open_grdecl(filename, keywords=[keyword]) as kw_generator:
        try:
            _, result = next(kw_generator)
        except StopIteration as si:
            raise ValueError(
                f"Cannot import {keyword}, not present in file {filename}?"
            ) from si

    # The values are stored in F order in the grdecl file
    f_order_values = np.asarray(result, dtype=dtype)
    return np.ascontiguousarray(f_order_values.reshape(dimensions, order="F"))


def import_bgrdecl(
    file_path: Union[str, os.PathLike[str]],
    field_name: str,
    dimensions: Tuple[int, int, int],
) -> Optional[npt.NDArray[np.float32]]:
    field_name = field_name.strip()
    with open(file_path, "rb") as f:
        for entry in ecl_data_io.lazy_read(f):
            keyword = str(entry.read_keyword()).strip()
            if keyword == field_name:
                values = entry.read_array()
                if np.issubdtype(values.dtype, np.integer):
                    values = values.astype(np.int32)
                else:
                    values = values.astype(np.float32)
                return values.reshape(dimensions, order="F")  # type: ignore
    return None


# pylint: disable=too-many-arguments
def export_grdecl(
    values: Union[
        np.ma.MaskedArray[Any, np.dtype[np.float32]], npt.NDArray[np.float32]
    ],
    file_path: Union[str, os.PathLike[str]],
    param_name: str,
    binary: bool = False,
    discrete: bool = False,
    dtype: npt.DTypeLike = None,
    fmt: Optional[str] = None,
) -> None:
    """Export ascii or binary GRDECL"""
    values = values.flatten(order="F")
    if np.ma.isMaskedArray(values):  # type: ignore
        values = values.filled(np.nan)  # type: ignore

    if binary:
        with open(file_path, "wb") as fh:
            if dtype is not None:
                ecl_data_io.write(fh, [(param_name.ljust(8), values.astype(dtype))])
            elif discrete:
                ecl_data_io.write(fh, [(param_name.ljust(8), values.astype(np.int32))])
            else:
                ecl_data_io.write(
                    fh, [(param_name.ljust(8), values.astype(np.float32))]
                )

    else:
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(param_name + "\n")
            for i, v in enumerate(values):
                fh.write(" ")
                if fmt:
                    fh.write(fmt % v)
                elif discrete:
                    fh.write(str(v))
                else:
                    fh.write(f"{v:3e}")
                if i % 6 == 5:
                    fh.write("\n")

            fh.write(" /\n")
