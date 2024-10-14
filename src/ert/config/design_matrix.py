from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from ert.config.gen_kw_config import GenKwConfig, TransformFunctionDefinition

from ._option_dict import option_dict
from .parsing import ConfigValidationError, ErrorInfo

if TYPE_CHECKING:
    from ert.config import ParameterConfig

DESIGN_MATRIX_GROUP = "DESIGN_MATRIX"


@dataclass
class DesignMatrix:
    xls_filename: Path
    design_sheet: str
    default_sheet: str

    def __post_init__(self) -> None:
        try:
            (
                self.active_realizations,
                self.design_matrix_df,
                self.parameter_configuration,
            ) = self.read_design_matrix()
        except (ValueError, AttributeError) as exc:
            raise ConfigValidationError.with_context(
                f"Error reading design matrix {self.xls_filename}: {exc}",
                str(self.xls_filename),
            ) from exc

    @classmethod
    def from_config_list(cls, config_list: list[str]) -> DesignMatrix:
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
                    "DESIGN_SHEET and DEFAULT_SHEET can not point to the same sheet."
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

    def merge_with_existing_parameters(
        self, existing_parameters: list[ParameterConfig]
    ) -> tuple[list[ParameterConfig], ParameterConfig | None]:
        """
        This method merges the design matrix parameters with the existing parameters and
        returns the new list of existing parameters, wherein we drop GEN_KW group having a full overlap with the design matrix group.
        GEN_KW group that was dropped will acquire a new name from the design matrix group.
        Additionally, the ParameterConfig which is the design matrix group is returned separately.

        Args:
            existing_parameters (List[ParameterConfig]): List of existing parameters

        Raises:
            ConfigValidationError: If there is a partial overlap between the design matrix group and any existing GEN_KW group

        Returns:
            tuple[List[ParameterConfig], ParameterConfig]: List of existing parameters and the dedicated design matrix group
        """

        new_param_config: list[ParameterConfig] = []

        design_parameter_group = self.parameter_configuration[DESIGN_MATRIX_GROUP]
        design_keys = []
        if isinstance(design_parameter_group, GenKwConfig):
            design_keys = [e.name for e in design_parameter_group.transform_functions]

        design_group_added = False
        for parameter_group in existing_parameters:
            if not isinstance(parameter_group, GenKwConfig):
                new_param_config += [parameter_group]
                continue
            existing_keys = [e.name for e in parameter_group.transform_functions]
            if set(existing_keys) == set(design_keys):
                if design_group_added:
                    raise ConfigValidationError(
                        "Multiple overlapping groups with design matrix found in existing parameters!\n"
                        f"{design_parameter_group.name} and {parameter_group.name}"
                    )

                design_parameter_group.name = parameter_group.name
                design_group_added = True
            elif set(design_keys) & set(existing_keys):
                raise ConfigValidationError(
                    "Overlapping parameter names found in design matrix!\n"
                    f"{DESIGN_MATRIX_GROUP}:{design_keys}\n{parameter_group.name}:{existing_keys}"
                    "\nThey need to much exactly or not at all."
                )
            else:
                new_param_config += [parameter_group]
        return new_param_config, design_parameter_group

    def read_design_matrix(
        self,
    ) -> tuple[list[bool], pd.DataFrame, dict[str, ParameterConfig]]:
        # Read the parameter names (first row) as strings to prevent pandas from modifying them.
        # This ensures that duplicate or empty column names are preserved exactly as they appear in the Excel sheet.
        # By doing this, we can properly validate variable names, including detecting duplicates or missing names.
        param_names = (
            pd.read_excel(
                self.xls_filename,
                sheet_name=self.design_sheet,
                nrows=1,
                header=None,
                dtype="string",
            )
            .iloc[0]
            .apply(lambda x: x.strip() if isinstance(x, str) else x)
        )
        design_matrix_df = DesignMatrix._read_excel(
            self.xls_filename,
            self.design_sheet,
            header=None,
            skiprows=1,
        )
        design_matrix_df.columns = param_names.to_list()

        if "REAL" in design_matrix_df.columns:
            if not is_integer_dtype(design_matrix_df.dtypes["REAL"]) or any(
                design_matrix_df["REAL"] < 0
            ):
                raise ValueError("REAL column must only contain positive integers")
            design_matrix_df = design_matrix_df.set_index(
                "REAL", drop=True, verify_integrity=True
            )

        if error_list := DesignMatrix._validate_design_matrix(design_matrix_df):
            error_msg = "\n".join(error_list)
            raise ValueError(f"Design matrix is not valid, error:\n{error_msg}")

        defaults_to_use = DesignMatrix._read_defaultssheet(
            self.xls_filename, self.default_sheet, design_matrix_df.columns.to_list()
        )
        design_matrix_df = design_matrix_df.assign(**defaults_to_use)

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
        reals = design_matrix_df.index.tolist()
        return (
            [x in reals for x in range(max(reals) + 1)],
            design_matrix_df,
            parameter_configuration,
        )

    @staticmethod
    def _read_excel(
        file_name: Path | str,
        sheet_name: str,
        usecols: list[int] | None = None,
        header: int | None = 0,
        skiprows: int | None = None,
        dtype: str | None = None,
    ) -> pd.DataFrame:
        """
        Reads an Excel file into a DataFrame, with options to filter columns and rows,
        and automatically drops columns that contain only NaN values.
        """
        df = pd.read_excel(
            io=file_name,
            sheet_name=sheet_name,
            usecols=usecols,
            header=header,
            skiprows=skiprows,
            dtype=dtype,
        )
        return df.dropna(axis=1, how="all")

    @staticmethod
    def _validate_design_matrix(design_matrix: pd.DataFrame) -> list[str]:
        """
        Validate user inputted design matrix
        :raises: ValueError if design matrix contains empty headers or empty cells
        """
        if design_matrix.empty:
            return []
        errors = []
        column_na_mask = design_matrix.columns.isna()
        column_indexes_unnamed = [
            index for index, value in enumerate(column_na_mask) if value
        ]
        if len(column_indexes_unnamed) > 0:
            errors.append(
                f"Column headers not present in column {column_indexes_unnamed}"
            )
        if not design_matrix.columns[~column_na_mask].is_unique:
            errors.append("Duplicate parameter names found in design sheet")
        empties = [
            f"Realization {design_matrix.index[i]}, column {design_matrix.columns[j]}"
            for i, j in zip(*np.where(pd.isna(design_matrix)), strict=False)
        ]
        if len(empties) > 0:
            errors.append(f"Design matrix contains empty cells {empties}")
        return errors

    @staticmethod
    def _read_defaultssheet(
        xls_filename: Path | str,
        defaults_sheetname: str,
        existing_parameters: list[str],
    ) -> dict[str, str | float]:
        """
        Construct a dict of keys and values to be used as defaults from the
        first two columns in a spreadsheet. Only returns the keys that are
        different from the exisiting parameters.

        Returns a dict of default values

        :raises: ValueError if defaults sheet is non-empty but non-parsable
        """
        default_df = DesignMatrix._read_excel(
            xls_filename,
            defaults_sheetname,
            header=None,
            dtype="string",
        )
        if default_df.empty:
            return {}
        if len(default_df.columns) < 2:
            raise ValueError("Defaults sheet must have at least two columns")
        empty_cells = [
            f"Row {default_df.index[i]}, column {default_df.columns[j]}"
            for i, j in zip(*np.where(pd.isna(default_df)), strict=False)
        ]
        if len(empty_cells) > 0:
            raise ValueError(f"Default sheet contains empty cells {empty_cells}")
        default_df[0] = default_df[0].apply(lambda x: x.strip())
        if not default_df[0].is_unique:
            raise ValueError("Default sheet contains duplicate parameter names")

        return {
            row[0]: convert_to_numeric(row[1])
            for _, row in default_df.iterrows()
            if row[0] not in existing_parameters
        }


def convert_to_numeric(x: str) -> str | float:
    try:
        return pd.to_numeric(x)
    except ValueError:
        return x
