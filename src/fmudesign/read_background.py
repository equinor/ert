from typing import Any

import pandas as pd

from .read_correlations import parse_sensitivity_correlations
from .utils import _has_value, _is_int, find_sheet


def read_background(inp_filename: str, bck_sheet: str) -> dict[str, Any]:
    """Reads excel sheet with background parameters and distributions

    Args:
        inp_filename (str): name of Excel workbook
        bck_sheet (str): name of sheet with background parameters

    Returns:
        dict with parameter names and distributions
    """
    backdict: dict[str, Any] = {}
    paramdict: dict[str, Any] = {}
    with pd.ExcelFile(inp_filename, engine="openpyxl") as workbook:
        sheet_names = [str(name) for name in workbook.sheet_names]
    try:
        bck_sheet = find_sheet(bck_sheet, names=sheet_names)
    except ValueError as err:
        raise ValueError(
            f"Sheet {bck_sheet!r} with background parameters, specified in the "
            f"general input sheet, was not found in {inp_filename!r}.\n"
            f"Sheets in workbook: {sheet_names}\n"
            "Use 'None' as background in the general input sheet if no "
            "background parameters are wanted."
        ) from err
    bck_input = (
        pd.read_excel(inp_filename, bck_sheet, engine="openpyxl")
        .dropna(axis=0, how="all")
        .loc[:, lambda df: ~df.columns.astype(str).str.contains("^Unnamed")]
    )

    backdict["correlations"] = None
    if "corr_sheet" in bck_input:
        backdict["correlations"] = parse_sensitivity_correlations(
            bck_input, inp_filename, group_description=f"background sheet {bck_sheet!r}"
        )

    for col_name in ("dist_param1", "dist_param2", "dist_param3", "dist_param4"):
        if col_name not in bck_input:
            bck_input[col_name] = float("NaN")

    for row in bck_input.itertuples():
        if not _has_value(row.param_name):
            raise ValueError(
                "Background parameters specified "
                "where one line has empty parameter "
                "name "
            )
        if not _has_value(row.dist_param1):
            raise ValueError(
                f"Parameter {row.param_name} has been input "
                "in background sheet but with empty "
                "first distribution parameter "
            )
        if not _has_value(row.dist_param2) and _has_value(row.dist_param3):
            raise ValueError(
                f"Parameter {row.param_name} has been input in "
                "background sheet with "
                'value for "dist_param3" while '
                '"dist_param2" is empty. This is not '
                "allowed"
            )
        if not _has_value(row.dist_param3) and _has_value(row.dist_param4):
            raise ValueError(
                f"Parameter {row.param_name} has been input in "
                "background sheet with "
                'value for "dist_param4" while '
                '"dist_param3" is empty. This is not '
                "allowed"
            )
        distparams = [
            item
            for item in [
                row.dist_param1,
                row.dist_param2,
                row.dist_param3,
                row.dist_param4,
            ]
            if _has_value(item)
        ]
        if "corr_sheet" in bck_input:
            corrsheet = None if not _has_value(row.corr_sheet) else row.corr_sheet
        else:
            corrsheet = None
        paramdict[str(row.param_name)] = [str(row.dist_name), distparams, corrsheet]
    backdict["parameters"] = paramdict

    if "decimals" in bck_input:
        decimals: dict[str, Any] = {}
        for row in bck_input.itertuples():
            if _has_value(row.decimals) and _is_int(row.decimals):
                decimals[row.param_name] = int(row.decimals)
        backdict["decimals"] = decimals

    return backdict
