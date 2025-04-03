from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from .parsing import ConfigValidationError, ErrorInfo
from .scalar_parameter import (
    DataSource,
    ScalarParameter,
    ScalarParameters,
    TransRawSettings,
)

if TYPE_CHECKING:
    pass

DESIGN_MATRIX_GROUP = "DESIGN_MATRIX"

from ert.shared.status.utils import convert_to_numeric


@dataclass
class DesignMatrix:
    xls_filename: Path
    design_sheet: str
    default_sheet: str | None

    def __post_init__(self) -> None:
        try:
            (
                self.active_realizations,
                self.design_matrix_df,
                self.scalars,
            ) = self.read_and_validate_design_matrix()
        except (ValueError, AttributeError) as exc:
            raise ConfigValidationError.with_context(
                f"Error reading design matrix {self.xls_filename}: {exc}",
                str(self.xls_filename),
            ) from exc

    @classmethod
    def from_config_list(cls, config_list: list[str | dict[str, str]]) -> DesignMatrix:
        filename = Path(cast(str, config_list[0]))
        options = cast(dict[str, str], config_list[1])
        design_sheet = options.get("DESIGN_SHEET", "DesignSheet")
        default_sheet = options.get("DEFAULT_SHEET", None)
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
        if design_sheet is not None and design_sheet == default_sheet:
            errors.append(
                ErrorInfo(
                    "DESIGN_SHEET and DEFAULT_SHEET can not point to the same sheet."
                ).set_context(config_list)
            )
        if errors:
            raise ConfigValidationError.from_collected(errors)
        assert design_sheet is not None
        return cls(
            xls_filename=filename,
            design_sheet=design_sheet,
            default_sheet=default_sheet,
        )

    def merge_with_other(self, dm_other: DesignMatrix) -> None:
        errors = []
        if self.active_realizations != dm_other.active_realizations:
            errors.append(
                ErrorInfo(
                    f"Design Matrices '{self.xls_filename.name} ({self.design_sheet} {self.default_sheet})' and "
                    f"'{dm_other.xls_filename.name} ({dm_other.design_sheet} {dm_other.default_sheet})' do not "
                    "have the same active realizations!"
                )
            )

        common_keys = set(self.design_matrix_df.columns) & set(
            dm_other.design_matrix_df.columns
        )
        non_identical_cols = set()
        if common_keys:
            for key in common_keys:
                if not self.design_matrix_df[key].equals(
                    dm_other.design_matrix_df[key]
                ):
                    non_identical_cols.add(key)
            if non_identical_cols:
                errors.append(
                    ErrorInfo(
                        f"Design Matrices '{self.xls_filename.name} ({self.design_sheet} {self.default_sheet})' and "
                        f"'{dm_other.xls_filename.name} ({dm_other.design_sheet} {dm_other.default_sheet})' "
                        f"contains non identical columns with the same name: {non_identical_cols}!"
                    )
                )
            errors.append(
                ErrorInfo(f"Design Matrices do not have unique keys {common_keys}!")
            )

        try:
            self.design_matrix_df = pd.concat(
                [self.design_matrix_df, dm_other.design_matrix_df], axis=1
            )
        except ValueError as exc:
            errors.append(ErrorInfo(f"Error when merging design matrices {exc}!"))

        for param in dm_other.scalars:
            if param.param_name not in common_keys:
                self.scalars.append(param)

        if errors:
            raise ConfigValidationError.from_collected(errors)

        try:
            self.design_matrix_df = pd.concat(
                [
                    self.design_matrix_df,
                    dm_other.design_matrix_df.drop(list(common_keys), axis=1),
                ],
                axis=1,
            )
        except ValueError as exc:
            raise ConfigValidationError(
                f"Error when merging design matrices '{self.xls_filename.name} ({self.design_sheet} {self.default_sheet})' and "
                f"'{dm_other.xls_filename.name} ({dm_other.design_sheet} {dm_other.default_sheet})': {exc}!"
            ) from exc

    def merge_with_existing_parameters(
        self, existing_scalars: ScalarParameters
    ) -> ScalarParameters:
        """
        This method merges the design matrix parameters with the existing parameters and
        returns the new list of existing parameters.
        Args:
            existing_scalars (ScalarParameters): existing scalar parameters


        Returns:
            ScalarParameters: new set of ScalarParameters
        """

        all_params: list[ScalarParameter] = []

        overlap_set = set()
        for existing_parameter in existing_scalars.scalars:
            if existing_parameter.input_source == DataSource.DESIGN_MATRIX:
                continue
            overlap = False
            for parameter_design in self.scalars:
                if existing_parameter.param_name == parameter_design.param_name:
                    parameter_design.group_name = existing_parameter.group_name
                    parameter_design.template_file = existing_parameter.template_file
                    parameter_design.output_file = existing_parameter.output_file
                    all_params.append(parameter_design)
                    overlap = True
                    overlap_set.add(existing_parameter.param_name)
                    break
            if not overlap:
                all_params.append(existing_parameter)

        for parameter_design in self.scalars:
            if parameter_design.param_name not in overlap_set:
                all_params.append(parameter_design)

        return ScalarParameters(scalars=all_params)

    def read_and_validate_design_matrix(
        self,
    ) -> tuple[list[bool], pd.DataFrame, list[ScalarParameter]]:
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
        design_matrix_df = pd.read_excel(
            io=self.xls_filename,
            sheet_name=self.design_sheet,
            header=None,
            skiprows=1,
        ).dropna(axis=1, how="all")
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
            raise ValueError(f"Design matrix is not valid, error(s):\n{error_msg}")

        if self.default_sheet is not None:
            defaults_to_use = DesignMatrix._read_defaultssheet(
                self.xls_filename,
                self.default_sheet,
                design_matrix_df.columns.to_list(),
            )
            default_df = pd.DataFrame(
                {
                    k: [v] * len(design_matrix_df.index)
                    for k, v in defaults_to_use.items()
                },
                index=design_matrix_df.index,
            )

            design_matrix_df = pd.concat([design_matrix_df, default_df], axis=1)

        scalars: list[ScalarParameter] = []
        for parameter in design_matrix_df.columns:
            scalars.append(
                ScalarParameter(
                    param_name=parameter,
                    group_name=DESIGN_MATRIX_GROUP,
                    input_source=DataSource.DESIGN_MATRIX,
                    distribution=TransRawSettings(),
                    template_file=None,
                    output_file=None,
                    update=False,
                )
            )

        reals = design_matrix_df.index.tolist()
        return (
            [x in reals for x in range(max(reals) + 1)],
            design_matrix_df,
            scalars,
        )

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
        if not design_matrix.columns[~column_na_mask].is_unique:
            errors.append("Duplicate parameter names found in design sheet")
        empties = [
            f"Realization {design_matrix.index[i]}, column {design_matrix.columns[j]}"
            for i, j in zip(*np.where(pd.isna(design_matrix)), strict=False)
        ]
        if len(empties) > 0:
            errors.append(f"Design matrix contains empty cells {empties}")

        for column_num, param_name in enumerate(design_matrix.columns):
            if pd.isna(param_name) or len(param_name.split()) == 0:
                errors.append(f"Empty parameter name found in column {column_num}.")
            elif len(param_name.split()) > 1:
                errors.append(
                    f"Multiple words in parameter name found in column {column_num} ({param_name})."
                )
            elif param_name.isnumeric():
                errors.append(f"Numeric parameter name found in column {column_num}.")
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
        default_df = pd.read_excel(
            io=xls_filename,
            sheet_name=defaults_sheetname,
            header=None,
            dtype="string",
        ).dropna(axis=1, how="all")
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
