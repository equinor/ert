from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import pandas as pd

from ert.config.gen_kw_config import GenKwConfig, TransformFunctionDefinition

from ._option_dict import option_dict
from .parsing import (
    ConfigValidationError,
    ErrorInfo,
)

if TYPE_CHECKING:
    from ert.config import (
        ParameterConfig,
    )

DESIGN_MATRIX_GROUP = "DESIGN_MATRIX"


@dataclass
class DesignMatrix:
    xls_filename: Path
    design_sheet: str
    default_sheet: str
    design_matrix_df: Optional[pd.DataFrame] = None
    parameter_configuration: Optional[dict[str, ParameterConfig]] = None

    @classmethod
    def from_config_list(cls, config_list: List[str]) -> "DesignMatrix":
        filename = Path(config_list[0])
        options = option_dict(config_list, 1)
        design_sheet = options.get("DESIGN_SHEET")
        default_sheet = options.get("DEFAULT_SHEET")
        errors = []
        if filename.suffix not in {
            ".xlsx",
            ".xls",
        }:
            errors.append(
                ErrorInfo(
                    f"DESIGN_MATRIX must be of format .xls or .xlsx; is '{filename}'"
                ).set_context(config_list)
            )
        if design_sheet is None:
            errors.append(
                ErrorInfo("Missing required DESIGN_SHEET").set_context(config_list)
            )
        if default_sheet is None:
            errors.append(
                ErrorInfo("Missing required DEFAULT_SHEET").set_context(config_list)
            )
        if design_sheet is not None and design_sheet == default_sheet:
            errors.append(
                ErrorInfo(
                    "DESIGN_SHEET and DEFAULT_SHEET can not be the same."
                ).set_context(config_list)
            )
        if errors:
            raise ConfigValidationError.from_collected(errors)
        assert design_sheet is not None
        assert default_sheet is not None
        return cls(
            xls_filename=filename,
            design_sheet=design_sheet,
            default_sheet=default_sheet,
        )

    def read_design_matrix(
        self,
    ) -> None:
        """
        Reads out all file content from different files and create dataframes
        """
        design_matrix_df = DesignMatrix._read_excel(
            self.xls_filename, self.design_sheet
        )
        if "REAL" in design_matrix_df.columns:
            design_matrix_df = design_matrix_df.set_index("REAL", drop=True)
        try:
            DesignMatrix._validate_design_matrix_header(design_matrix_df)
        except ValueError as err:
            raise ValueError(f"Design matrix not valid, error: {err!s}") from err

        # Todo: Check for invalid realizations, drop them maybe?
        # This should probably handle/(fill in) missing values in design_matrix_sheet as well? Or maybe not.
        defaults = DesignMatrix._read_defaultssheet(
            self.xls_filename, self.default_sheet
        )
        for k, v in defaults.items():
            if k not in design_matrix_df.columns:
                design_matrix_df[k] = v

        # ignoring errors here is deprecated in pandas, should find another solution
        # design_matrix_sheet = design_matrix_sheet.apply(pd.to_numeric, errors="ignore")
        parameter_configuration: dict[str, ParameterConfig] = {}
        transform_function_definitions: list[TransformFunctionDefinition] = []
        for parameter in design_matrix_df.columns:
            transform_function_definitions.append(
                TransformFunctionDefinition(
                    name=parameter,
                    param_name="RAW",
                    values=[],
                )
            )
        parameter_configuration[DESIGN_MATRIX_GROUP] = GenKwConfig(
            name=DESIGN_MATRIX_GROUP,
            forward_init=False,
            template_file=None,
            output_file=None,
            transform_function_definitions=transform_function_definitions,
            update=False,
        )

        design_matrix_df.columns = pd.MultiIndex.from_product(
            [[DESIGN_MATRIX_GROUP], design_matrix_df.columns]
        )
        self.design_matrix_df = design_matrix_df
        self.parameter_configuration = parameter_configuration

    @staticmethod
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
            unnamed = design_matrix.loc[
                :, design_matrix.columns.str.contains("^Unnamed")
            ]
        except ValueError as err:
            # We catch because int/floats as column headers
            # in xlsx gets read as int/float and is not valid to index by.
            raise ValueError(
                f"Invalid value in design matrix header, error: {err !s}"
            ) from err
        column_indexes = [int(x.split(":")[1]) for x in unnamed.columns.to_numpy()]
        if len(column_indexes) > 0:
            raise ValueError(f"Column headers not present in column {column_indexes}")

    @staticmethod
    def _read_defaultssheet(
        xlsfilename: Path | str, defaultssheetname: str
    ) -> dict[str, str]:
        """
        Construct a dataframe of keys and values to be used as defaults from the
        first two columns in a spreadsheet.

        Returns a dict of default values

        :raises: ValueError if defaults sheet is non-empty but non-parsable
        """
        default_df = DesignMatrix._read_excel(
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

        return {row[0]: row[1] for _, row in default_df.iterrows()}
