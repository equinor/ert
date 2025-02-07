from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

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

    def merge_with_other(self, dm_other: DesignMatrix) -> None:
        errors = []
        if self.active_realizations != dm_other.active_realizations:
            errors.append(
                ErrorInfo("Design Matrices don't have the same active realizations!")
            )

        common_keys = set(
            self.design_matrix_df.select(pl.exclude("REAL")).columns
        ) & set(dm_other.design_matrix_df.select(pl.exclude("REAL")).columns)
        if common_keys:
            errors.append(
                ErrorInfo(f"Design Matrices do not have unique keys {common_keys}!")
            )

        if errors:
            raise ConfigValidationError.from_collected(errors)
        try:
            self.design_matrix_df = pl.concat(
                [
                    self.design_matrix_df,
                    dm_other.design_matrix_df.select(pl.exclude("REAL")),
                ],
                how="horizontal",
            )
        except ValueError as exc:
            raise ConfigValidationError.from_info(
                ErrorInfo(f"Error when merging design matrices {exc}!")
            ) from exc

        for tfd in dm_other.parameter_configuration.transform_function_definitions:
            self.parameter_configuration.transform_function_definitions.append(tfd)

    def merge_with_existing_parameters(
        self, existing_parameters: list[ParameterConfig]
    ) -> tuple[list[ParameterConfig], GenKwConfig]:
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

        design_parameter_group = self.parameter_configuration
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
    ) -> tuple[list[bool], pl.DataFrame, GenKwConfig]:
        # Read the parameter names (first row) as strings to prevent pandas from modifying them.
        # This ensures that duplicate or empty column names are preserved exactly as they appear in the Excel sheet.
        # By doing this, we can properly validate variable names, including detecting duplicates or missing names.
        try:
            param_names = (
                pl.read_excel(
                    self.xls_filename,
                    sheet_name=self.design_sheet,
                    has_header=False,
                    read_options={"n_rows": 1, "dtypes": "string"},
                )
                .select(pl.all().str.strip_chars())
                .row(0)
            )
        except pl.exceptions.NoDataError as err:
            raise ValueError("Design sheet is empty.") from err

        design_matrix_df = pl.read_excel(
            self.xls_filename,
            sheet_name=self.design_sheet,
            has_header=False,
            drop_empty_cols=True,
            drop_empty_rows=True,
            raise_if_empty=False,
            read_options={"skip_rows": 1},
        )

        if error_list := DesignMatrix._validate_design_matrix(
            design_matrix_df, param_names
        ):
            error_msg = "\n".join(error_list)
            raise ValueError(f"Design matrix is not valid, error(s):\n{error_msg}")
        design_matrix_df.columns = param_names
        if "REAL" in design_matrix_df.columns:
            if (
                not design_matrix_df.schema.get("REAL").is_integer()
                or design_matrix_df.get_column("REAL").lt(0).any()
                or design_matrix_df.get_column("REAL").is_duplicated().any()
            ):
                raise ValueError(
                    "REAL column must only contain unique positive integers"
                )

        else:
            design_matrix_df = design_matrix_df.with_row_index(name="REAL")

        defaults_to_use = DesignMatrix._read_defaultssheet(
            self.xls_filename, self.default_sheet, design_matrix_df.columns
        )
        design_matrix_df = design_matrix_df.with_columns(
            pl.lit(value).alias(name) for name, value in defaults_to_use.items()
        )

        transform_function_definitions: list[TransformFunctionDefinition] = []
        for parameter in design_matrix_df.select(pl.exclude("REAL")).columns:
            transform_function_definitions.append(
                TransformFunctionDefinition(
                    name=parameter,
                    param_name="RAW",
                    values=[],
                )
            )
        parameter_configuration = GenKwConfig(
            name=DESIGN_MATRIX_GROUP,
            forward_init=False,
            template_file=None,
            output_file=None,
            transform_function_definitions=transform_function_definitions,
            update=False,
        )

        reals = design_matrix_df.get_column("REAL").to_list()
        return (
            [x in reals for x in range(max(reals) + 1)],
            design_matrix_df,
            parameter_configuration,
        )

    @staticmethod
    def _validate_design_matrix(
        design_matrix: pl.DataFrame, column_names: tuple[str]
    ) -> list[str]:
        """
        Validate user inputted design matrix
        :raises: ValueError if design matrix contains empty headers or empty cells
        """
        if design_matrix.is_empty():
            return []
        errors = []
        param_name_count = Counter(p for p in column_names if p is not None)
        duplicate_param_names = [(n, c) for n, c in param_name_count.items() if c > 1]
        if duplicate_param_names:
            duplicates_formatted = ", ".join(
                f"{name}({count})" for name, count in duplicate_param_names
            )
            errors.append(
                f"Duplicate parameter names found in design sheet: {duplicates_formatted}"
            )
        empties = [
            f"Row {i}, column {j}"
            for i, j in zip(
                *np.where(design_matrix.select(pl.all().is_null())), strict=False
            )
        ]
        if len(empties) > 0:
            errors.append(f"Design matrix contains empty cells {empties}")

        for column_num, param_name in enumerate(column_names):
            if param_name is None or len(param_name.split()) == 0:
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
        xls_filename: Path,
        defaults_sheetname: str,
        existing_parameters: list[str],
    ) -> dict[str, str | float | int]:
        """
        Construct a dict of keys and values to be used as defaults from the
        first two columns in a spreadsheet. Only returns the keys that are
        different from the exisiting parameters.

        Returns a dict of default values

        :raises: ValueError if defaults sheet is non-empty but non-parsable
        """

        default_df = pl.read_excel(
            xls_filename,
            sheet_name=defaults_sheetname,
            has_header=False,
            drop_empty_cols=True,
            drop_empty_rows=True,
            raise_if_empty=False,
            read_options={"dtypes": "string"},
        )
        if default_df.is_empty():
            return {}
        if len(default_df.columns) < 2:
            raise ValueError("Defaults sheet must have at least two columns")
        empty_cells = [
            f"Row {i}, column {j}"
            for i, j in zip(
                *np.where(default_df.select(pl.all().is_null())), strict=False
            )
        ]
        if len(empty_cells) > 0:
            raise ValueError(f"Default sheet contains empty cells {empty_cells}")
        default_df = default_df.with_columns(pl.nth(0).str.strip_chars())
        if default_df.select(pl.nth(0)).is_duplicated().any():
            raise ValueError("Default sheet contains duplicate parameter names")

        return {
            row[0]: convert_to_numeric(row[1])
            for row in default_df.iter_rows()
            if row[0] not in existing_parameters
        }


def convert_to_numeric(x: str) -> str | float | int:
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)

        except ValueError:
            return x
