from typing import Any, Literal

import pandas as pd

from .utils import _has_value


def parse_distribution_parameters(
    rows: pd.DataFrame, *, source: Literal["sensitivity", "background"]
) -> dict[str, list[Any]]:
    for column in ("dist_param1", "dist_param2", "dist_param3", "dist_param4"):
        if column not in rows:
            rows[column] = float("NaN")

    location = "in background sheet" if source == "background" else 'as type "dist"'
    gap_location = " in background sheet" if source == "background" else ""
    parameters: dict[str, list[Any]] = {}

    for row in rows.itertuples():
        if not _has_value(row.param_name):
            description = (
                "Background parameters"
                if source == "background"
                else f"Dist sensitivity {row.sensname}"
            )
            raise ValueError(
                f"{description} specified where one line has empty parameter name "
            )
        if not _has_value(row.dist_param1):
            raise ValueError(
                f"Parameter {row.param_name} has been input "
                f"{location} but with empty first distribution parameter "
            )
        if not _has_value(row.dist_param2) and _has_value(row.dist_param3):
            raise ValueError(
                f"Parameter {row.param_name} has been input{gap_location} with "
                'value for "dist_param3" while "dist_param2" is empty. '
                "This is not allowed"
            )
        if not _has_value(row.dist_param3) and _has_value(row.dist_param4):
            raise ValueError(
                f"Parameter {row.param_name} has been input{gap_location} with "
                'value for "dist_param4" while "dist_param3" is empty. '
                "This is not allowed"
            )

        arguments = [
            value
            for value in (
                row.dist_param1,
                row.dist_param2,
                row.dist_param3,
                row.dist_param4,
            )
            if _has_value(value)
        ]
        correlation_sheet = getattr(row, "corr_sheet", None)
        if not _has_value(correlation_sheet):
            correlation_sheet = None

        parameters[str(row.param_name)] = [
            str(row.dist_name),
            arguments,
            correlation_sheet,
        ]

    return parameters
