from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype

from ert.config.gen_kw_config import (
    DataSource,
    GenKwConfig,
    TransformFunctionDefinition,
)
from ert.config.parameter_config import ParameterConfig
from ert.shared.status.utils import convert_to_numeric

from .parsing import ConfigValidationError, ErrorInfo

DESIGN_MATRIX_GROUP = "DESIGN_MATRIX"


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
                self.parameter_configuration,
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
                    f"Design Matrices '{self.xls_filename.name} ({self.design_sheet} "
                    f"{self.default_sheet})' and '{dm_other.xls_filename.name} "
                    f"({dm_other.design_sheet} {dm_other.default_sheet})' do not "
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
                        f"Design Matrices '{self.xls_filename.name} "
                        f"({self.design_sheet} {self.default_sheet})' and "
                        f"'{dm_other.xls_filename.name} ({dm_other.design_sheet} "
                        f"{dm_other.default_sheet})' "
                        "contains non identical columns with the same name: "
                        f"{non_identical_cols}!"
                    )
                )

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
                f"Error when merging design matrices "
                f"'{self.xls_filename.name} ({self.design_sheet} {self.default_sheet})'"
                f" and '{dm_other.xls_filename.name} ({dm_other.design_sheet} "
                f"{dm_other.default_sheet})': {exc}!"
            ) from exc

        for tfd in dm_other.parameter_configuration.transform_function_definitions:
            if tfd.name not in common_keys:
                self.parameter_configuration.transform_function_definitions.append(tfd)

    def merge_with_existing_parameters(
        self, existing_parameters: list[ParameterConfig]
    ) -> list[ParameterConfig]:
        return [
            config
            if not isinstance(config, GenKwConfig)
            else self._merge_with_single_gen_kw(config)
            for config in existing_parameters
        ]

    def _merge_with_single_gen_kw(self, gen_kw_config: GenKwConfig) -> GenKwConfig:
        """
        This method merges the design matrix parameters with a single GenKwConfig.
        Returns:
            new merged GenKwConfig
        """

        all_params: list[TransformFunctionDefinition] = []

        overlap_set = set()
        for existing_parameter in gen_kw_config.transform_function_definitions:
            if existing_parameter.input_source == DataSource.DESIGN_MATRIX:
                continue
            overlap = False
            for (
                parameter_design
            ) in self.parameter_configuration.transform_function_definitions:
                if existing_parameter.name == parameter_design.name:
                    parameter_design.group_name = existing_parameter.group_name
                    all_params.append(parameter_design)
                    overlap = True
                    overlap_set.add(existing_parameter.param_name)
                    break
            if not overlap:
                all_params.append(existing_parameter)

        for (
            parameter_design
        ) in self.parameter_configuration.transform_function_definitions:
            if parameter_design.name not in overlap_set:
                all_params.append(parameter_design)

        return GenKwConfig(transform_function_definitions=all_params)

    def read_and_validate_design_matrix(
        self,
    ) -> tuple[list[bool], pd.DataFrame, GenKwConfig]:
        # Read the parameter names (first row) as strings to prevent pandas from
        # modifying them. This ensures that duplicate or empty column names are
        # preserved exactly as they appear in the Excel sheet. By doing this, we
        # can properly validate variable names, including detecting duplicates or
        # missing names.
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
        )
        design_matrix_df.columns = param_names.to_list()
        design_matrix_df = design_matrix_df.dropna(axis=1, how="all")

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

        transform_function_definitions: list[TransformFunctionDefinition] = []
        for parameter in design_matrix_df.columns:
            transform_function_definitions.append(
                TransformFunctionDefinition(
                    name=parameter,
                    param_name="RAW",
                    group_name=DESIGN_MATRIX_GROUP,
                    values=[],
                    input_source=DataSource.DESIGN_MATRIX,
                    update=False,
                )
            )
        parameter_configuration = GenKwConfig(
            name=DESIGN_MATRIX_GROUP,
            transform_function_definitions=transform_function_definitions,
        )

        reals = design_matrix_df.index.tolist()
        return (
            [x in reals for x in range(max(reals) + 1)],
            design_matrix_df,
            parameter_configuration,
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
                    "Multiple words in parameter name found in column "
                    f"{column_num} ({param_name})."
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
        default_df = default_df.iloc[:, 0:2]
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
