from collections import defaultdict
from typing import List

import pandas as pd
import xarray as xr

from ert.config.gen_kw_config import GenKwConfig, TransformFunctionDefinition


def read_design_matrix(
    ert_config,
    xlsfilename,
    designsheetname="DesignSheet01",
    defaultssheetname="DefaultValues",
):
    # pylint: disable=too-many-arguments
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
        raise ValueError(f"Design matrix not valid, error: {str(err)}") from err

    # Todo: Check for invalid realizations, drop them maybe?

    if designsheetname == defaultssheetname:
        raise ValueError("Design-sheet and defaults-sheet can not be the same")

    defaults = _read_defaultssheet(xlsfilename, defaultssheetname)
    for k, v in defaults.items():
        if k not in design_matrix_sheet.columns:
            design_matrix_sheet[k] = v
    design_matrix_sheet = design_matrix_sheet.apply(pd.to_numeric, errors="ignore")

    parameter_groups = defaultdict(list)
    parameter_map = []
    for param in design_matrix_sheet.columns:
        try:
            # Try to match the parameter name to existing parameter group
            parameter_name = next(
                val.name
                for val in ert_config.ensemble_config.parameter_configuration
                if param in val
            )
        except StopIteration:
            parameter_name = "DESIGN_MATRIX"
        parameter_groups[parameter_name].append(param)
        parameter_map.append((parameter_name, param))
    design_matrix_sheet.columns = pd.MultiIndex.from_tuples(parameter_map)
    return design_matrix_sheet


def initialize_parameters(design_matrix_sheet, storage, ert_config, exp_name, ens_name):
    existing_parameters = ert_config.ensemble_config.parameter_configs
    parameter_configs = []
    for parameter_group in design_matrix_sheet.columns.get_level_values(0).unique():
        parameters = design_matrix_sheet[parameter_group].columns
        transform_function_definitions: List[TransformFunctionDefinition] = []
        for param in parameters:
            transform_function_definitions.append(
                TransformFunctionDefinition(
                    name=param,
                    param_name="RAW",
                    values=[],
                )
            )
        existing = existing_parameters.get(parameter_group)
        parameter_configs.append(
            GenKwConfig(
                name=parameter_group,
                forward_init=False,
                template_file=existing.template_file
                if isinstance(existing, GenKwConfig)
                else None,
                output_file=existing.output_file
                if isinstance(existing, GenKwConfig)
                else None,
                transform_function_definitions=transform_function_definitions,
                update=False,
            )
        )

    experiment = storage.create_experiment(
        parameters=parameter_configs,
        responses=ert_config.ensemble_config.response_configuration,
        observations=ert_config.observations,
        name=exp_name,
    )
    ensemble = storage.create_ensemble(
        experiment,
        name=ens_name,
        ensemble_size=max(design_matrix_sheet.index),
    )
    for i in range(len(design_matrix_sheet)):
        for parameter_group in experiment.parameter_configuration:
            row = design_matrix_sheet.iloc[i][parameter_group]
            ds = xr.Dataset(
                {
                    "values": ("names", list(row.values)),
                    "transformed_values": ("names", list(row.values)),
                    "names": list(row.keys()),
                }
            )
            ensemble.save_parameters(parameter_group, i, ds)
    return ensemble


def _read_excel(file_name, sheet_name, usecols=None, header=0):
    """
    Make dataframe from excel file
    :return: Dataframe
    :raises: OsError if file not found
    :raises: ValueError if file not loaded correctly
    """
    dframe = pd.read_excel(
        file_name,
        sheet_name,
        dtype=str,
        usecols=usecols,
        header=header,
    )
    return dframe.dropna(axis=1, how="all")


def _validate_design_matrix_header(design_matrix):
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
            f"Invalid value in design matrix header, error: {str(err)}"
        ) from err
    column_indexes = [int(x.split(":")[1]) for x in unnamed.columns.values]
    if len(column_indexes) > 0:
        raise ValueError(f"Column headers not present in column {column_indexes}")


def _read_defaultssheet(xlsfilename, defaultssheetname):
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
                    (
                        f'Parameter name "{paramname}" in default values contains '
                        "initial or trailing whitespace."
                    )
                )

    else:
        return {}

    default_df.rename(columns={0: "keys", 1: "defaults"}, inplace=True)
    defaults = {}
    for _, row in default_df.iterrows():
        defaults[row["keys"]] = row["defaults"]
    return defaults
