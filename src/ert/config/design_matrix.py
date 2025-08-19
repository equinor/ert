from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl
from polars.exceptions import InvalidOperationError

from .gen_kw_config import GenKwConfig, TransformFunctionDefinition
from .parsing import ConfigValidationError, ErrorInfo

if TYPE_CHECKING:
    from ert.config import ParameterConfig

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
                f"Error reading design matrix {self.xls_filename}"
                f" ({self.design_sheet} {self.default_sheet or ''}):"
                f" {exc}",
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
                    f"{self.default_sheet or ''})' and '{dm_other.xls_filename.name} "
                    f"({dm_other.design_sheet} {dm_other.default_sheet or ''})' do not "
                    "have the same active realizations!"
                )
            )

        common_keys = set(
            self.design_matrix_df.select(pl.exclude("realization")).columns
        ) & set(dm_other.design_matrix_df.columns)
        non_identical_cols = set()
        if common_keys:
            for key in common_keys:
                if not self.design_matrix_df.select(key).equals(
                    dm_other.design_matrix_df.select(key)
                ):
                    non_identical_cols.add(key)
            if non_identical_cols:
                errors.append(
                    ErrorInfo(
                        f"Design Matrices '{self.xls_filename.name} "
                        f"({self.design_sheet} {self.default_sheet or ''})' and "
                        f"'{dm_other.xls_filename.name} ({dm_other.design_sheet} "
                        f"{dm_other.default_sheet or ''})' "
                        "contains non identical columns with the same name: "
                        f"{non_identical_cols}!"
                    )
                )

        if errors:
            raise ConfigValidationError.from_collected(errors)

        try:
            self.design_matrix_df = pl.concat(
                [
                    self.design_matrix_df,
                    dm_other.design_matrix_df.select(
                        pl.exclude([*list(common_keys), "realization"])
                    ),
                ],
                how="horizontal",
            )
        except ValueError as exc:
            raise ConfigValidationError(
                f"Error when merging design matrices "
                f"'{self.xls_filename.name} ({self.design_sheet}"
                f" {self.default_sheet or ''})'"
                f" and '{dm_other.xls_filename.name} ({dm_other.design_sheet} "
                f"{dm_other.default_sheet or ''})': {exc}!"
            ) from exc

        for tfd in dm_other.parameter_configuration.transform_function_definitions:
            if tfd.name not in common_keys:
                self.parameter_configuration.transform_function_definitions.append(tfd)

    def merge_with_existing_parameters(
        self, existing_parameters: list[ParameterConfig]
    ) -> tuple[list[ParameterConfig], GenKwConfig]:
        """
        This method merges the design matrix parameters with the existing parameters and
        returns the new list of existing parameters, wherein we drop GEN_KW group having
        a full overlap with the design matrix group. GEN_KW group that was dropped will
        acquire a new name from the design matrix group. Additionally, the
        ParameterConfig which is the design matrix group is returned separately.

        Args:
            existing_parameters (List[ParameterConfig]): List of existing parameters

        Raises:
            ConfigValidationError: If there is a partial overlap between the design
            matrix group and any existing GEN_KW group

        Returns:
            tuple[List[ParameterConfig], ParameterConfig]: List of existing parameters
            and the dedicated design matrix group
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
                        "Multiple overlapping groups with design matrix found in "
                        "existing parameters!\n"
                        f"{design_parameter_group.name} and {parameter_group.name}"
                    )

                design_parameter_group.name = parameter_group.name
                design_group_added = True
            elif set(design_keys) & set(existing_keys):
                raise ConfigValidationError(
                    "Overlapping parameter names found in design matrix!\n"
                    f"{DESIGN_MATRIX_GROUP}:{design_keys}\n{parameter_group.name}:{existing_keys}"
                    "\nThey need to match exactly or not at all."
                )
            else:
                new_param_config += [parameter_group]
        return new_param_config, design_parameter_group

    def read_and_validate_design_matrix(
        self,
    ) -> tuple[list[bool], pl.DataFrame, GenKwConfig]:
        # Read the parameter names (first row) as strings to prevent polars from
        # modifying them. This ensures that duplicate or empty column names are
        # preserved exactly as they appear in the Excel sheet. By doing this, we
        # can properly validate variable names, including detecting duplicates or
        # missing names.
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
            raise ValueError("Design sheet headers are empty.") from err
        design_matrix_df = (
            pl.read_excel(
                self.xls_filename,
                sheet_name=self.design_sheet,
                has_header=False,
                drop_empty_cols=False,
                drop_empty_rows=True,
                raise_if_empty=False,
                infer_schema_length=None,
                read_options={"skip_rows": 1},
            )
            .with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))
            .with_columns(pl.col(pl.String).str.strip_chars())
        )
        if design_matrix_df.is_empty():
            raise ValueError("Design sheet body is empty.")
        string_cols = [
            col for col, dtype in design_matrix_df.schema.items() if dtype == pl.String
        ]
        design_matrix_df = design_matrix_df.with_columns(
            [
                pl.when(
                    pl.col(col).str.to_lowercase().is_in(["nan", "null", "none", ""])
                )
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
                for col in string_cols
            ]
        )
        # We drop the columns that are empty and have an empty parameter name
        columns_to_keep = [
            i
            for i, s in enumerate(design_matrix_df)
            if s.null_count() != design_matrix_df.height or param_names[i]
        ]

        design_matrix_df = design_matrix_df.select(
            design_matrix_df.columns[i] for i in columns_to_keep
        )
        param_names = tuple(param_names[i] for i in columns_to_keep)

        if errors := DesignMatrix._validate_design_matrix(
            design_matrix_df, param_names
        ):
            error_msg = "\n".join(errors)
            raise ValueError(f"Design matrix is not valid, error(s):\n{error_msg}")

        design_matrix_df.columns = list(param_names)

        if self.default_sheet is not None:
            defaults_to_use = DesignMatrix._read_defaultssheet(
                self.xls_filename, self.default_sheet, design_matrix_df.columns
            )
            design_matrix_df = design_matrix_df.with_columns(
                pl.lit(value).alias(name) for name, value in defaults_to_use.items()
            )

        if "realization" in design_matrix_df.schema:
            raise ValueError(
                "'realization' is a reserved internal keyword in ERT"
                " and cannot be used as a parameter name."
            )
        if "REAL" in design_matrix_df.schema:
            design_matrix_df = design_matrix_df.rename({"REAL": "realization"})
            real_dt = design_matrix_df.schema.get("realization")
            assert real_dt is not None
            if (
                not real_dt.is_integer()
                or (
                    design_matrix_df.get_column("realization").lt(0)
                    | design_matrix_df.get_column("realization").is_duplicated()
                ).any()
            ):
                raise ValueError(
                    "REAL column must only contain unique positive integers"
                )
        else:
            design_matrix_df = design_matrix_df.with_row_index(name="realization")

        design_matrix_df = convert_numeric_string_columns(design_matrix_df)
        transform_function_definitions = [
            TransformFunctionDefinition(name=col, param_name="RAW", values=[])
            for col in design_matrix_df.columns
            if col != "realization"
        ]
        parameter_configuration = GenKwConfig(
            name=DESIGN_MATRIX_GROUP,
            forward_init=False,
            transform_function_definitions=transform_function_definitions,
            update=False,
        )

        reals = design_matrix_df.get_column("realization").to_list()
        return (
            [x in reals for x in range(max(reals) + 1)],
            design_matrix_df,
            parameter_configuration,
        )

    @staticmethod
    def _validate_design_matrix(
        design_matrix: pl.DataFrame, param_names: tuple[str]
    ) -> list[str]:
        """
        Validate user inputted design matrix
        :raises: ValueError if design matrix contains empty headers or empty cells
        """
        errors = []
        param_name_count = Counter(p for p in param_names if p is not None)
        duplicate_param_names = [(n, c) for n, c in param_name_count.items() if c > 1]
        if duplicate_param_names:
            duplicates_formatted = ", ".join(
                f"{name}({count})" for name, count in duplicate_param_names
            )
            errors.append(
                "Duplicate parameter names found in design sheet:"
                f" {duplicates_formatted}"
            )
        empties = [
            f"Row {i}, column {param_names[j]}"
            for i, j in zip(
                *np.where(design_matrix.select(pl.all().is_null())),
                strict=False,
            )
        ]
        if len(empties) > 0:
            errors.append(f"Design matrix contains empty cells {empties}")

        for column_num, param_name in enumerate(param_names):
            if param_name is None or len(param_name.split()) == 0:
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
        default_df = default_df.select(pl.nth(0, 1)).with_columns(
            pl.nth(0, 1).str.strip_chars()
        )
        default_df = default_df.with_columns(
            [
                pl.when(
                    pl.col(col).str.to_lowercase().is_in(["nan", "null", "none", ""])
                )
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
                for col in default_df.columns
            ]
        )
        empty_cells = [
            f"Row {i}, column {j}"
            for i, j in zip(
                *np.where(default_df.select(pl.all().is_null())), strict=False
            )
        ]
        if len(empty_cells) > 0:
            raise ValueError(f"Default sheet contains empty cells {empty_cells}")
        if default_df.select(pl.nth(0)).is_duplicated().any():
            raise ValueError("Default sheet contains duplicate parameter names")

        return {
            row[0]: convert_to_numeric(row[1])
            for row in default_df.iter_rows()
            if row[0] not in existing_parameters
        }


def convert_numeric_string_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Automatically convert string columns to numeric (int or float) where possible"""
    for col, dtype in zip(df.columns, df.dtypes, strict=False):
        if dtype == pl.String:
            try:
                df = df.with_columns(pl.col(col).cast(pl.Int64, strict=True).alias(col))
                continue
            except InvalidOperationError:
                pass

            try:  # noqa: SIM105
                df = df.with_columns(
                    pl.col(col).cast(pl.Float64, strict=True).alias(col)
                )
            except InvalidOperationError:
                pass

    return df


def convert_to_numeric(x: str) -> str | float | int:
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)

        except ValueError:
            return x
