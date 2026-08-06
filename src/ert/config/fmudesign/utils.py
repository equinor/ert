"""
Module for utility functions that do not belong elsewhere.
"""

from typing import Any

import pandas as pd


def parameters_from_extern(filename: str) -> pd.DataFrame:
    """Read parameter values or background values
    from specified file. Format either Excel ('xlsx')
    or csv.

    Args:
        filename (str): name of file
    """
    if str(filename).endswith(".xlsx"):
        return (
            pd.read_excel(filename, engine="openpyxl")
            .dropna(axis=0, how="all")
            .loc[:, lambda df: ~df.columns.str.contains("^Unnamed")]
        )

    if str(filename).endswith(".csv"):
        return pd.read_csv(filename)

    raise ValueError(
        "External file with parameter values should "
        "be on Excel or csv format "
        "and end with .xlsx or .csv"
    )


def seeds_from_extern(filename: str) -> list[int]:
    """Read parameter values or background values
    from specified file. Format either Excel ('xlsx')
    or csv.

    Args:
        filename (str): name of file
    """
    if str(filename).endswith(".xlsx"):
        df_seeds = (
            pd.read_excel(filename, header=None, engine="openpyxl")
            .dropna(axis=0, how="all")
            .dropna(axis=1, how="all")
        )
        return df_seeds.iloc[:, 0].tolist()

    if str(filename).endswith(".csv") or str(filename).endswith(".txt"):
        df_seeds = pd.read_csv(filename, header=None)
        return df_seeds.iloc[:, 0].tolist()

    raise ValueError(
        "External file with seed values should "
        "be on Excel or csv format "
        "and end with .xlsx .csv or .txt"
    )


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
        "semeio.fmudesign Warning:                                     \n"
        "Using designinput sheets where "
        "corr_sheet is only specified for one parameter "
        "will cause non-correlated parameters .\n"
        f"ONLY ONE PARAMETER WAS SPECIFIED TO USE CORR_SHEET {corr_group_name}\n"
        "\n"
        "Note change in how correlated parameters are specified \n"
        "from fmudeisgn version 1.0.1 in August 2019 :\n"
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


def to_numeric_safe(val: float | str) -> int | float | str:
    """Convert all values that CAN be converted to numeric. Retain the rest.
    This used to be pd.to_numeric(..., errors='ignore'), but was deprecated.

    Examples
    --------
    >>> df = pd.DataFrame({'a': ['cat', '3.5', '-1', 0, 'dog']})
    >>> df.map(to_numeric_safe).a.values
    array(['cat', np.float64(3.5), np.int64(-1), 0, 'dog'], dtype=object)

    >>> [to_numeric_safe(e) for e in [5, '3', 'dog']]
    [5, np.int64(3), 'dog']

    """
    assert not isinstance(val, pd.Series | pd.DataFrame | list)
    try:
        return pd.to_numeric(val)  # noq
    except (ValueError, TypeError):
        return val


def map_dependencies(
    df: pd.DataFrame, *, dependencies: dict[str, Any], verbose: bool = False
) -> pd.DataFrame:
    """Return a new copy of `df` with dependencies mapped.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3, 4], 'b': ['A', 'B', 'C', 'D']})
    >>> dependencies = {'a': {'from_values': [1, 2, 3, 4],
    ...                       'to_params':{'c': [1, 4, 9, 16]}}}
    >>> map_dependencies(df, dependencies=dependencies)
       a  b   c
    0  1  A   1
    1  2  B   4
    2  3  C   9
    3  4  D  16

    A messy mix of numbers and strings:

    >>> df = pd.DataFrame({'a': ['1', '2', 3, 4], 'b': ['A', 'B', 'C', 'D']})
    >>> dependencies = {'a': {'from_values': ['1', 2, '3', 4],
    ...                       'to_params':{'c': [1, 4, 9, '16']}}}
    >>> map_dependencies(df, dependencies=dependencies)
       a  b   c
    0  1  A   1
    1  2  B   4
    2  3  C   9
    3  4  D  16

    If no `to_params` are given, then the `from` column is copied:

    >>> dependencies = {'a': {'from_values': ['1', 2, '3', 4],
    ...                       'to_params':{'c': [1, 4, 9, '16'],
    ...                                    'd': []}}}
    >>> map_dependencies(df, dependencies=dependencies)
       a  b   c  d
    0  1  A   1  1
    1  2  B   4  2
    2  3  C   9  3
    3  4  D  16  4
    """

    df = df.copy()
    for from_param, from_dict in dependencies.items():
        # No column to map from
        if from_param not in df.columns:
            continue

        from_values = from_dict["from_values"]
        from_values = [to_numeric_safe(value) for value in from_values]

        for to_param, to_values_ in from_dict["to_params"].items():
            to_values = [to_numeric_safe(value) for value in to_values_]

            # No values to map to => to_param = copy(from_param)
            if not to_values:
                df = df.assign(**{to_param: df[from_param].map(to_numeric_safe)})
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
            not_mapped = set(df[from_param].map(to_numeric_safe)) - set(from_values)
            if not_mapped:
                msg = (
                    f"Mapping dependencies {from_param!r} to {to_param!r} using "
                    f"mapping:\n{mapping!r}\n failed. The following values could "
                    f"not be mapped:\n{not_mapped!r}"
                )
                raise ValueError(msg)

            df = df.assign(
                **{
                    # Bind loop variables as default args to the lambda
                    to_param: lambda df, from_param=from_param, mapping=mapping: (
                        df[from_param].map(to_numeric_safe).map(mapping)
                    )
                }
            )
            if verbose:
                print(
                    f"Mapping dependency. From {from_param!r} "
                    f"to {to_param!r} using map:"
                )
                for from_, to_ in mapping.items():
                    print(f" {from_} => {to_}")

    return df
