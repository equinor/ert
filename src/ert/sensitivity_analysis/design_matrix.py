from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from ert.config.gen_kw_config import GenKwConfig, TransformFunctionDefinition

if TYPE_CHECKING:
    from ert.config import (
        ErtConfig,
    )
    from ert.storage import LocalEnsemble, LocalStorage

DESIGN_MATRIX_GROUP = "DESIGN_MATRIX"


def read_design_matrix(
    ert_config: ErtConfig,
    xlsfilename: Path | str,
    designsheetname: str = "DesignSheet01",
    defaultssheetname: str = "DefaultValues",
) -> pd.DataFrame:
    """
    Reads out all file content from different files and create dataframes
    """
    design_matrix_sheet = _read_excel(xlsfilename, designsheetname)
    if "REAL" in design_matrix_sheet.columns:
        design_matrix_sheet.set_index(design_matrix_sheet["REAL"])
        del design_matrix_sheet["REAL"]
    try:
        _validate_design_matrix_header(design_matrix_sheet)
    except ValueError as err:
        raise ValueError(f"Design matrix not valid, error: {err!s}") from err

    # Todo: Check for invalid realizations, drop them maybe?

    if designsheetname == defaultssheetname:
        raise ValueError("Design-sheet and defaults-sheet can not be the same")

    # This should probably handle/(fill in) missing values in design_matrix_sheet as well
    defaults = _read_defaultssheet(xlsfilename, defaultssheetname)
    for k, v in defaults.items():
        if k not in design_matrix_sheet.columns:
            design_matrix_sheet[k] = v

    # ignoring errors here is deprecated in pandas, should find another solution
    # design_matrix_sheet = design_matrix_sheet.apply(pd.to_numeric, errors="ignore")

    existing_parameters = {
        param.name
        for param_group in ert_config.ensemble_config.parameter_configuration
        if isinstance(param_group, GenKwConfig)
        for param in param_group.transform_function_definitions
    }
    intersect = existing_parameters.intersection(set(design_matrix_sheet.columns))
    # This errors if parameters exists already, this behaviour should be discussed.
    if intersect:
        msg = "The following parameters were specified both"
        f"as gen_kw and in the design matrix: {intersect}"
        raise ValueError(msg)
    return design_matrix_sheet


def initialize_parameters(
    design_matrix_sheet: pd.DataFrame,
    storage: LocalStorage,
    ert_config: ErtConfig,
    exp_name: str,
    ens_name: str,
) -> LocalEnsemble:
    existing_parameters = ert_config.ensemble_config.parameter_configuration
    parameters = design_matrix_sheet.columns
    transform_function_definitions: list[TransformFunctionDefinition] = []
    for param in parameters:
        transform_function_definitions.append(
            TransformFunctionDefinition(
                name=param,
                param_name="RAW",
                values=[],
            )
        )
    existing_parameters.append(
        GenKwConfig(
            name=DESIGN_MATRIX_GROUP,
            forward_init=False,
            template_file=None,
            output_file=None,
            transform_function_definitions=transform_function_definitions,
            update=False,
        )
    )

    experiment = storage.create_experiment(
        parameters=existing_parameters,
        responses=ert_config.ensemble_config.response_configuration,
        observations=ert_config.observations,
        name=exp_name,
    )
    ensemble = storage.create_ensemble(
        experiment,
        name=ens_name,
        ensemble_size=len(design_matrix_sheet.index),
    )
    for i in design_matrix_sheet.index:
        row: pd.Series = design_matrix_sheet.iloc[i]
        ds = xr.Dataset(
            {
                "values": ("names", list(row.to_numpy())),
                "transformed_values": ("names", list(row.to_numpy())),
                "names": list(row.keys()),
            }
        )
        ensemble.save_parameters(DESIGN_MATRIX_GROUP, i, ds)
    return ensemble


def _read_excel(
    file_name: Path | str,
    sheet_name: str,
    usecols: int | list[int] | None = None,
    header: int | None = 0,
) -> pd.DataFrame:
    """
    Make dataframe from excel file
    :return: Dataframe
    :raises: OsError if file not found
    :raises: ValueError if file not loaded correctly
    """
    dframe: pd.DataFrame = pd.read_excel(
        file_name,
        sheet_name,
        usecols=usecols,
        header=header,
    )
    return dframe.dropna(axis=1, how="all")


def _validate_design_matrix_header(design_matrix: pd.DataFrame) -> None:
    """
    Validate header in user inputted design matrix
    :raises: ValueError if design matrix contains empty headers
    """
    if design_matrix.empty:
        return
    try:
        unnamed = design_matrix.loc[:, design_matrix.columns.str.contains("^Unnamed")]
    except ValueError as err:
        # We catch because int/floats as column headers
        # in xlsx gets read as int/float and is not valid to index by.
        raise ValueError(
            f"Invalid value in design matrix header, error: {err !s}"
        ) from err
    column_indexes = [int(x.split(":")[1]) for x in unnamed.columns.to_numpy()]
    if len(column_indexes) > 0:
        raise ValueError(f"Column headers not present in column {column_indexes}")


def _read_defaultssheet(
    xlsfilename: Path | str, defaultssheetname: str
) -> dict[str, str]:
    """
    Construct a dataframe of keys and values to be used as defaults from the
    first two columns in a spreadsheet.

    Returns a dict of default values

    :raises: ValueError if defaults sheet is non-empty but non-parsable
    """
    if defaultssheetname:
        default_df = _read_excel(
            xlsfilename, defaultssheetname, usecols=[0, 1], header=None
        )
        if default_df.empty:
            return {}
        if len(default_df.columns) < 2:
            raise ValueError("Defaults sheet must have at least two columns")
        # Look for initial or trailing whitespace in parameter names. This
        # is disallowed as it can create user confusion and has no use-case.
        for paramname in default_df.loc[:, 0]:
            if paramname != paramname.strip():
                raise ValueError(
                    f'Parameter name "{paramname}" in default values contains '
                    "initial or trailing whitespace."
                )

    else:
        return {}

    default_df = default_df.rename(columns={0: "keys", 1: "defaults"})
    defaults = {}
    for _, row in default_df.iterrows():
        defaults[row["keys"]] = row["defaults"]
    return defaults
