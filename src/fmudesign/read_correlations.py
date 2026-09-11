from collections import defaultdict
from typing import Any

import pandas as pd

from .design_distributions import read_correlations
from .utils import _has_value


def parse_sensitivity_correlations(
    sensgroup: pd.DataFrame, inputfile: str, group_description: str | None = None
) -> dict[str, Any] | None:
    """Parse correlation information from a sensitivity group.

    Args:
        sensgroup: rows describing the parameters, either a sensitivity group
            from the designinput sheet or the background sheet.
        inputfile: name of the Excel workbook holding the correlation sheets.
        group_description: how to refer to `sensgroup` in error messages.
            Defaults to the sensname of the group.
    """

    # No correlation sheet column exists
    if "corr_sheet" not in sensgroup.columns:
        return None

    # The column exists, but it is all blank
    if sensgroup["corr_sheet"].dropna().empty:
        return None

    if group_description is None:
        group_description = f"sensitivity group {sensgroup['sensname'].iloc[0]!r}"

    correlations: dict[str, Any] = {"inputfile": inputfile}

    # Create a mapping 'corr_to_params' like:
    # {'corr1': ['var_A', 'var_B', ...], ...}
    corr_to_params = defaultdict(list)
    for _, row in sensgroup.iterrows():
        if not _has_value(row["corr_sheet"]):
            continue
        corr_to_params[row["corr_sheet"]].append(row["param_name"])

    # Open the correlation sheet and peek at it
    # We want to verify that if variables ['A', 'B'] point to the corr sheet,
    # then exactly those variables are also defined in the sheet
    for corr_sheet, parameters in corr_to_params.items():
        df_corr = read_correlations(excel_filename=inputfile, corr_sheet=corr_sheet)
        if set(df_corr.columns) != set(parameters):
            msg = f"Mismatch between parameters in {group_description} "
            msg += f"pointing to\ncorrelation sheet {corr_sheet!r} and "
            msg += "parameters specified in that correlation sheet.\n"
            msg += f"Parameters in {group_description}: {sorted(set(parameters))}\n"
            msg += f"Parameters in correlation sheet: {sorted(set(df_corr.columns))}\n"
            msg += "These parameters must be specified one-to-one."
            raise ValueError(msg)

    correlations["sheetnames"] = list(corr_to_params)

    return correlations
