"""Module for utility functions that do not belong elsewhere."""

import csv
import math
import re
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from python_calamine import CalamineWorkbook

from ert.config.design_matrix import convert_to_numeric

_NUMERIC_STRING = re.compile(
    r"[+-]?(?:([0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|inf(?:inity)?)",
    re.IGNORECASE,
)
_NULL_VALUES = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
]


def parameter_series(name: str, values: Iterable[object]) -> pl.Series:
    values = [
        value.item() if isinstance(value, np.generic) else value for value in values
    ]
    values = [
        None if isinstance(value, float) and math.isnan(value) else value
        for value in values
    ]
    types = {type(value) for value in values if value is not None}
    dtype = pl.Object if len(types) > 1 and not types <= {int, float} else None
    return pl.Series(name, values, dtype=dtype, strict=False)


def _compatible_series(columns: Sequence[pl.Series]) -> list[pl.Series]:
    dtypes = {column.dtype for column in columns} - {pl.Null}
    if len(dtypes) <= 1 or all(dtype.is_numeric() for dtype in dtypes):
        return list(columns)
    # A common string dtype would erase native numeric and Boolean cell types.
    return [
        pl.Series(column.name, column.to_list(), dtype=pl.Object) for column in columns
    ]


def concat_design_frames(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    frames = list(frames)
    for name in dict.fromkeys(name for frame in frames for name in frame.columns):
        indices = [index for index, frame in enumerate(frames) if name in frame]
        columns = _compatible_series([frames[index][name] for index in indices])
        for index, column in zip(indices, columns, strict=True):
            frames[index] = frames[index].with_columns(column)
    return pl.concat(frames, how="diagonal_relaxed")


def fill_parameter_nulls(column: pl.Series, values: pl.Series) -> pl.Series:
    if column.dtype.is_float():
        column = column.fill_nan(None)
    elif column.dtype == pl.Object:
        column = parameter_series(column.name, column)
    column, values = _compatible_series([column, values])
    return column.fill_null(values)


def numeric_parameter_series(column: pl.Series) -> pl.Series:
    if column.dtype not in {pl.String, pl.Object}:
        return column
    return parameter_series(column.name, map(to_numeric_safe, column))


def read_excel_values(filename: str, sheet_name: str | None = None) -> pl.DataFrame:
    with CalamineWorkbook.from_path(filename) as workbook:
        sheet = workbook.get_sheet_by_name(
            workbook.sheet_names[0] if sheet_name is None else sheet_name
        )
        rows = sheet.to_python(skip_empty_area=False)
    if not rows:
        return pl.DataFrame()

    original_names = [
        str(value) if value else f"Unnamed: {index}"
        for index, value in enumerate(rows[0])
    ]
    names: list[str] = []
    for name in original_names:
        adjusted_name = name
        if name in names:
            suffix = 1
            while f"{name}.{suffix}" in names or f"{name}.{suffix}" in original_names:
                suffix += 1
            adjusted_name = f"{name}.{suffix}"
        names.append(adjusted_name)
    columns = (
        [
            parameter_series(
                name,
                (
                    None if isinstance(value, str) and value in _NULL_VALUES else value
                    for value in values
                ),
            )
            for name, values in zip(names, zip(*rows[1:], strict=True), strict=True)
        ]
        if len(rows) > 1
        else [pl.Series(name, [], dtype=pl.Null) for name in names]
    )
    return pl.DataFrame(columns).filter(~pl.all_horizontal(pl.all().is_null()))


def excel_sheet_names(filename: Path | str) -> list[str]:
    with CalamineWorkbook.from_path(filename) as workbook:
        return workbook.sheet_names


def parameters_from_extern(filename: str) -> pl.DataFrame:
    """Read parameter values or background values
    from specified file. Format either Excel ('xlsx')
    or csv. Blank CSV records are skipped; explicit empty fields are retained.

    Args:
        filename (str): name of file
    """
    if not Path(filename).is_file():
        raise ValueError(f"External file '{filename}' does not exist.")

    if str(filename).endswith(".xlsx"):
        values = read_excel_values(filename)
        return values.select(
            name for name in values.columns if not name.startswith("Unnamed")
        )

    if str(filename).endswith(".csv"):
        with Path(filename).open(encoding="utf-8-sig", newline="") as csv_file:
            lines = csv_file.readlines()
        reader = csv.reader(lines)
        records = []
        start_line = 0
        # Record boundaries preserve blank lines inside quoted fields.
        for _ in reader:
            record = "".join(lines[start_line : reader.line_num])
            if record.strip(" \t\r\n"):
                records.append(record)
            start_line = reader.line_num
        return pl.read_csv(
            StringIO("".join(records)),
            infer_schema_length=None,
            null_values=_NULL_VALUES,
        )

    raise ValueError(
        "External file with parameter values should "
        "be on Excel or csv format "
        "and end with .xlsx or .csv"
    )


def seeds_from_extern(filename: Path | str) -> list[int]:
    """Read integer seed values from the first column of an Excel ('xlsx')
    or csv/txt file. Blank cells and lines are skipped.

    Args:
        filename (str): name of file
    """
    if str(filename).endswith(".xlsx"):
        seeds = pl.read_excel(
            filename, has_header=False, read_options={"dtypes": "string"}
        ).to_series(0)
    elif str(filename).endswith((".csv", ".txt")):
        seeds = pl.read_csv(filename, has_header=False, infer_schema=False).to_series(0)
    else:
        raise ValueError(
            "External file with seed values should "
            "be on Excel or csv format "
            "and end with .xlsx .csv or .txt"
        )

    seeds = seeds.str.strip_chars().replace("", None).drop_nulls()
    try:
        return seeds.cast(pl.Int64).to_list()
    except pl.exceptions.InvalidOperationError as err:
        raise ValueError(
            f"Seed values in {str(filename)!r} must be integers: {err}"
        ) from err


def find_max_realisations(config: dict[str, Any]) -> int:
    """Finds the maximum number of realisations over all sensitivity cases."""
    max_reals = config.get("repeats", 0)
    for sens_info in config["sensitivities"].values():
        max_reals = max(sens_info.get("numreal", 0), max_reals)
    assert max_reals > 0
    return max_reals


def printwarning(corr_group_name: str) -> None:
    print(
        "#######################################################\n"
        "fmudesign Warning:                                     \n"
        "Using designinput sheets where "
        "corr_sheet is only specified for one parameter "
        "will cause non-correlated parameters .\n"
        f"ONLY ONE PARAMETER WAS SPECIFIED TO USE CORR_SHEET {corr_group_name}\n"
        "\n"
        "Note change in how correlated parameters are specified \n"
        "from fmudesign version 1.0.1 in August 2019:\n"
        "Name of correlation sheet must be specified for each "
        "parameter in correlation matrix. \n"
        "This to enable use of several correlation sheets. "
        "This also means non-correlated parameters do not "
        "have to be included in correlation matrix. \n "
        "See documentation: \n"
        "https://equinor.github.io/fmu-tools/"
        "fmudesign.html#create-design-matrix-for-"
        "one-by-one-sensitivities\n"
        "\n"
        "####################################################\n"
    )


def to_numeric_safe(val: object) -> object:
    """Convert numeric strings without changing categorical values.

    Examples
    --------
    >>> [to_numeric_safe(e) for e in [5, '3', 'dog']]
    [5, 3, 'dog']

    """
    if isinstance(val, str):
        if not val:
            return None
        if _NUMERIC_STRING.fullmatch(val.strip()):
            return convert_to_numeric(val)
    return val


def _raise_if_duplicates(container: Iterable[Hashable]) -> None:
    """Raises a descriptive error if there are duplicates in the container."""
    duplicates = {k: v for (k, v) in Counter(container).items() if v > 1}
    if duplicates:
        raise ValueError(f"Duplicates with counts: {duplicates}")


def map_dependencies(
    df: pl.DataFrame, *, dependencies: dict[str, Any], verbose: bool = False
) -> pl.DataFrame:
    """Return a new copy of `df` with dependencies mapped.

    Examples
    --------
    >>> df = pl.DataFrame({'a': [1, 2, 3, 4], 'b': ['A', 'B', 'C', 'D']})
    >>> dependencies = {'a': {'from_values': [1, 2, 3, 4],
    ...                       'to_params':{'c': [1, 4, 9, 16]}}}
    >>> map_dependencies(df, dependencies=dependencies).to_dict(as_series=False)
    {'a': [1, 2, 3, 4], 'b': ['A', 'B', 'C', 'D'], 'c': [1, 4, 9, 16]}

    A messy mix of numbers and strings:

    >>> df = pl.DataFrame([
    ...     pl.Series('a', ['1', '2', 3, 4], dtype=pl.Object),
    ...     pl.Series('b', ['A', 'B', 'C', 'D']),
    ... ])
    >>> dependencies = {'a': {'from_values': ['1', 2, '3', 4],
    ...                       'to_params':{'c': [1, 4, 9, '16']}}}
    >>> map_dependencies(df, dependencies=dependencies).to_dict(as_series=False)
    {'a': ['1', '2', 3, 4], 'b': ['A', 'B', 'C', 'D'], 'c': [1, 4, 9, 16]}

    If no `to_params` are given, then the `from` column is copied:

    >>> dependencies = {'a': {'from_values': ['1', 2, '3', 4],
    ...                       'to_params':{'c': [1, 4, 9, '16'],
    ...                                    'd': []}}}
    >>> map_dependencies(df, dependencies=dependencies).to_dict(
    ...     as_series=False
    ... )  # doctest: +NORMALIZE_WHITESPACE
    {'a': ['1', '2', 3, 4], 'b': ['A', 'B', 'C', 'D'], 'c': [1, 4, 9, 16],
     'd': [1, 2, 3, 4]}
    """

    df = df.clone()
    for from_param, from_dict in dependencies.items():
        # No column to map from
        if from_param not in df.columns:
            continue

        from_values = [to_numeric_safe(value) for value in from_dict["from_values"]]
        try:
            _raise_if_duplicates(from_values)
        except ValueError as err:
            raise ValueError(
                f"Duplicate dependency keys for {from_param!r}\n{err}"
            ) from err

        for to_param, to_values_ in from_dict["to_params"].items():
            to_values = [to_numeric_safe(value) for value in to_values_]
            source = [to_numeric_safe(value) for value in df[from_param]]

            # No values to map to => to_param = copy(from_param)
            if not to_values:
                df = df.with_columns(parameter_series(to_param, source))
                if verbose:
                    print(f"Copied {from_param!r} to {to_param!r}")
                continue

            if len(from_values) != len(to_values):
                msg = (
                    f"Mapping dependencies {from_param!r} to {to_param!r} failed.\n"
                    f"Length mismatch.\nMapping from values: {from_values!r}"
                    f"\nMapping to values: {to_values!r}"
                )
                raise ValueError(msg)

            # At this point we have a mapping 'from_param' - > 'to_param'
            # defined elementwise by values of 'from_values' -> 'to_values'
            mapping = dict(zip(from_values, to_values, strict=False))

            # Check that every value will be mapped
            not_mapped = set(source) - set(from_values)
            if not_mapped:
                msg = (
                    f"Mapping dependencies {from_param!r} to {to_param!r} using "
                    f"mapping:\n{mapping!r}\n failed. The following values could "
                    f"not be mapped:\n{not_mapped!r}"
                )
                raise ValueError(msg)

            df = df.with_columns(
                parameter_series(to_param, (mapping[value] for value in source))
            )
            if verbose:
                print(
                    f"Mapping dependency. From {from_param!r} "
                    f"to {to_param!r} using map:"
                )
                for from_, to_ in mapping.items():
                    print(f" {from_} => {to_}")

    return df


def find_sheet(name: str, names: list[str]) -> str:
    """Search for Excel sheets with a soft matching. Raises ValueError if zero
    or more than one match is found.

    Examples:
    >>> find_sheet('general_input', ['generalinput', 'designinput', 'defaultinput'])
    'generalinput'
    >>> find_sheet('variable_input', ['generalinput', 'designinput', 'defaultinput'])
    Traceback (most recent call last):
      ...
    ValueError: No match for variable_input: ['generalinput', 'designinput', 'defaultinput']
    """  # ruff: ignore[line-too-long]

    def sanitize(inputstring: str) -> str:
        return inputstring.lower().strip().replace("_", "")

    found = [name_i for name_i in names if sanitize(name) == sanitize(name_i)]
    if not found:
        raise ValueError(f"No match for {name}: {names}")
    if len(found) > 1:
        raise ValueError(f"More than one match for {name}: {found}")
    return found[0]


def _has_value(value: Any) -> bool:
    """Returns False only if the argument is np.nan"""
    try:
        return not np.isnan(value)
    except TypeError:
        return True


def _is_int(teststring: str) -> bool:
    """Test if string is a finite integer"""
    try:
        if not np.isnan(int(teststring)):
            return math.isclose((float(teststring) % 1), 0, abs_tol=1e-14)
    except ValueError:
        return False
    else:
        return False  # It was a "number", but it was NaN.


def resolve_path(target: str | None, *, base_file: str | None = None) -> str | None:
    """Try to resolve the path of potential file 'target' either as relative to
    'base_file' or cwd.
    If not, return target.
    """
    if target is None:
        return None

    if (
        base_file is not None
        and (relative_path := Path(base_file).parent / target).is_file()
    ):
        return str(relative_path.resolve())

    if (path := Path(target)).is_file():
        return str(path.resolve())

    return target
