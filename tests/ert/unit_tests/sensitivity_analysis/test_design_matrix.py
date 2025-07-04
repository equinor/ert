import numpy as np
import polars as pl
import pytest
from xlsxwriter import Workbook

from ert.config import (
    DESIGN_MATRIX_GROUP,
    DesignMatrix,
    GenKwConfig,
)
from ert.config.gen_kw_config import TransformFunctionDefinition
from tests.ert.conftest import _create_design_matrix


@pytest.mark.parametrize(
    "design_sheet_pd, default_sheet_pd, error_msg",
    [
        pytest.param(
            pl.DataFrame(
                {
                    "REAL": [0, 1, 2],
                    "c": [9, 10, 11.1],
                    "d": [0, 2, 0],
                },
                strict=False,
            ),
            pl.DataFrame([["e", 1]], orient="row"),
            "",
            id="ok_merge",
        ),
        pytest.param(
            pl.DataFrame(
                {
                    "a": [1, 2, 3],
                    "c": [9, 10, 11.1],
                    "d": [0, 2, 0],
                },
                strict=False,
            ),
            pl.DataFrame([["e", 1]], orient="row"),
            "",
            id="ok_merge_with_identical_columns",
        ),
        pytest.param(
            pl.DataFrame(
                {
                    "REAL": [0, 1, 2],
                    "a": [1, 2, 4],
                }
            ),
            pl.DataFrame([["e", 1]], orient="row"),
            (
                "Design Matrices .* and .* contains non "
                r"identical columns with the same name: \{'a'\}!"
            ),
            id="not_unique_keys",
        ),
        pytest.param(
            pl.DataFrame(
                {
                    "REAL": [0, 1],
                    "d": [1, 2],
                }
            ),
            pl.DataFrame([["e", 1]], orient="row"),
            r"Design Matrices .* and .* do not have the same active realizations!",
            id="not_same_acitve_realizations",
        ),
    ],
)
def test_merge_multiple_occurrences(
    tmp_path, design_sheet_pd, default_sheet_pd, error_msg
):
    _create_design_matrix(
        tmp_path / "design_matrix_1.xlsx",
        pl.DataFrame(
            {
                "REAL": [0, 1, 2],
                "a": [1, 2, 3],
                "b": [0, 2, 0],
                " ": ["", "", ""],
            },
        ),
        pl.DataFrame([["a", 1], ["b", 4]], orient="row"),
    )
    design_matrix_1 = DesignMatrix(
        tmp_path / "design_matrix_1.xlsx", "DesignSheet", "DefaultSheet"
    )
    _create_design_matrix(
        tmp_path / "design_matrix_2.xlsx", design_sheet_pd, default_sheet_pd
    )
    design_matrix_2 = DesignMatrix(
        tmp_path / "design_matrix_2.xlsx", "DesignSheet", "DefaultSheet"
    )

    if error_msg:
        with pytest.raises(ValueError, match=error_msg):
            design_matrix_1.merge_with_other(design_matrix_2)
    else:
        design_matrix_1.merge_with_other(design_matrix_2)
        design_params = design_matrix_1.parameter_configuration
        assert all(param in design_params for param in ("a", "b", "c", "d"))
        assert design_matrix_1.active_realizations == [True, True, True]
        df = design_matrix_1.design_matrix_df
        np.testing.assert_equal(df["a"], np.array([1, 2, 3]))
        np.testing.assert_equal(df["b"], np.array([0, 2, 0]))
        np.testing.assert_equal(df["c"], np.array([9, 10, 11.1]))
        np.testing.assert_equal(df["d"], np.array([0, 2, 0]))


@pytest.mark.parametrize(
    "parameters, error_msg",
    [
        pytest.param(
            {"COEFFS": ["a", "b"]},
            "",
            id="genkw_replaced",
        ),
        pytest.param(
            {"COEFFS": ["a"]},
            "Overlapping parameter names found in design matrix!",
            id="ValidationErrorOverlapping",
        ),
        pytest.param(
            {"COEFFS": ["aa", "bb"], "COEFFS2": ["cc", "dd"]},
            "",
            id="DESIGN_MATRIX_GROUP",
        ),
        pytest.param(
            {"COEFFS": ["a", "b"], "COEFFS2": ["a", "b"]},
            (
                "Multiple overlapping groups with design matrix "
                "found in existing parameters!"
            ),
            id="ValidationErrorMultipleGroups",
        ),
    ],
)
def test_read_and_merge_with_existing_parameters(tmp_path, parameters, error_msg):
    extra_genkw_config = []
    if parameters:
        for group_name in parameters:
            extra_genkw_config.append(
                GenKwConfig(
                    name=group_name,
                    forward_init=False,
                    transform_function_definitions=[
                        TransformFunctionDefinition(param, "UNIFORM", [0, 1])
                        for param in parameters[group_name]
                    ],
                    update=True,
                )
            )

    realizations = [0, 1, 2]
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pl.DataFrame(
        {
            "REAL": realizations,
            "a": [1, 2, 3],
            "b": [0, 2, 0],
        }
    )
    default_sheet_df = pl.DataFrame([["a", 1], ["b", 4]], orient="row")
    _create_design_matrix(design_path, design_matrix_df, default_sheet_df)
    design_matrix = DesignMatrix(design_path, "DesignSheet", "DefaultSheet")
    if error_msg:
        with pytest.raises(ValueError, match=error_msg):
            design_matrix.merge_with_existing_parameters(extra_genkw_config)
    elif len(parameters) == 1:
        new_config_parameters, design_group = (
            design_matrix.merge_with_existing_parameters(extra_genkw_config)
        )
        assert len(new_config_parameters) == 0
        assert design_group.name == "COEFFS"
    elif len(parameters) == 2:
        new_config_parameters, design_group = (
            design_matrix.merge_with_existing_parameters(extra_genkw_config)
        )
        assert len(new_config_parameters) == 2
        assert design_group.name == DESIGN_MATRIX_GROUP


def test_reading_design_matrix(tmp_path):
    realizations = [0, 1, 4]
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pl.DataFrame(
        {
            "REAL": realizations,
            "a": [1, 2, 3],
            "b": [0, 2, 0],
            "c": ["low", "high", "medium"],
        }
    )
    default_sheet_df = pl.DataFrame(
        [["one", 1, ""], ["b", 4, ""], ["d", "case_name", 3]], orient="row"
    )
    _create_design_matrix(design_path, design_matrix_df, default_sheet_df)
    design_matrix = DesignMatrix(design_path, "DesignSheet", "DefaultSheet")
    design_params = design_matrix.parameter_configuration
    assert all(param in design_params for param in ("a", "b", "c", "one", "d"))
    assert design_matrix.active_realizations == [True, True, False, False, True]


@pytest.mark.parametrize(
    "real_column, error_msg",
    [
        pytest.param(
            [0, 1, 1],
            "REAL column must only contain unique positive integers",
            id="duplicate entries",
        ),
        pytest.param(
            [0, 1.1, 2],
            "REAL column must only contain unique positive integers",
            id="invalid float values",
        ),
        pytest.param(
            [0, "a", 10],
            "REAL column must only contain unique positive integers",
            id="invalid types",
        ),
    ],
)
def test_reading_design_matrix_validate_reals(tmp_path, real_column, error_msg):
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pl.DataFrame(
        {
            "REAL": real_column,
            "a": [1, 2, 3],
            "b": [0, 2, 0],
            "c": [3, 1, 3],
        },
        strict=False,
    )
    default_sheet_df = pl.DataFrame()
    _create_design_matrix(design_path, design_matrix_df, default_sheet_df)

    with pytest.raises(ValueError, match=error_msg):
        DesignMatrix(design_path, "DesignSheet", "DefaultSheet")


@pytest.mark.parametrize(
    "column_names, error_msg",
    [
        pytest.param(
            ["a", "b", "a"],
            "Duplicate parameter names found in design sheet",
            id="duplicate entries",
        ),
        pytest.param(
            ["a   ", "b", "       a"],
            "Duplicate parameter names found in design sheet",
            id="duplicate entries with whitespaces",
        ),
        pytest.param(
            ["a", "b", "parameter name with spaces"],
            "Multiple words in parameter name found in column 2.",
            id="multiple words in parameter name",
        ),
        pytest.param(
            ["a", "b", " "],
            "Empty parameter name found in column 2",
            id="dataframe loads parameter name as whitespace",
        ),
        pytest.param(["a", "b", "3"], "Numeric parameter name found in column 2"),
        pytest.param(
            ["a", "b c d e", 33],
            (
                "Multiple words in parameter name found in column 1 "
                r"\(b c d e\)\.\nNumeric parameter name found in column 2"
            ),
            id="multiple errors",
        ),
    ],
)
def test_reading_design_matrix_validate_headers(tmp_path, column_names, error_msg):
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pl.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    default_sheet_df = pl.DataFrame([["one", 1], ["b", 4], ["d", 6]], orient="row")
    _create_design_matrix(design_path, design_matrix_df, default_sheet_df)

    with Workbook(design_path) as xl_write:
        design_matrix_df.write_excel(xl_write, worksheet="DesignSheet")
        default_sheet_df.write_excel(
            xl_write, worksheet="DefaultSheet", include_header=False
        )
        ws = xl_write.get_worksheet_by_name("DesignSheet")
        for col_idx, header in enumerate(column_names):
            ws.write(0, col_idx, header)
    with pytest.raises(ValueError, match=error_msg):
        DesignMatrix(design_path, "DesignSheet", "DefaultSheet")


@pytest.mark.parametrize(
    "values, error_msg",
    [
        pytest.param(
            [0, None, 1],
            r"Design matrix contains empty cells \['Row 1, column a'\]",
            id="duplicate entries",
        ),
        pytest.param(
            [0, "      ", 1],
            r"Design matrix contains empty cells \['Row 1, column a'\]",
            id="whitespace entries",
        ),
        pytest.param(
            [0, "some", np.nan],
            r"Design matrix contains empty cells \['Row 2, column a'\]",
            id="invalid float values",
        ),
    ],
)
def test_reading_design_matrix_validate_cells(tmp_path, values, error_msg):
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pl.DataFrame(
        {
            "REAL": [1, 5, 7],
            "a": values,
            "b": [0, 2, 0],
            "c": [3, 1, 3],
        },
        strict=False,
    )
    default_sheet_df = pl.DataFrame()
    _create_design_matrix(design_path, design_matrix_df, default_sheet_df)

    with pytest.raises(ValueError, match=error_msg):
        DesignMatrix(design_path, "DesignSheet", "DefaultSheet")


@pytest.mark.parametrize(
    "data, error_msg",
    [
        pytest.param(
            [["one"], ["b"], ["d"]],
            "Defaults sheet must have at least two columns",
            id="Too few columns",
        ),
        pytest.param(
            [["one", 1], ["b", ""], ["d", 6]],
            r"Default sheet contains empty cells \['Row 1, column 1'\]",
            id="empty cells",
        ),
        pytest.param(
            [["something", 1], ["b", "          "], ["d", 6]],
            r"Default sheet contains empty cells \['Row 1, column 1'\]",
            id="whitespace entries",
        ),
        pytest.param(
            [["something", 1], ["b", "None"], ["d", 6]],
            r"Default sheet contains empty cells \['Row 1, column 1'\]",
            id="None entries",
        ),
        pytest.param(
            [[" a", 1], ["a ", "some"], ["d", 6]],
            r"Default sheet contains duplicate parameter names",
            id="duplicate parameter names",
        ),
        pytest.param(
            [["realization", 1], ["a ", "some"], ["d", 6]],
            r"'realization' is a reserved internal keyword in ERT"
            " and cannot be used as a parameter name.",
            id="realization in default values",
        ),
    ],
)
def test_reading_default_sheet_validation(tmp_path, data, error_msg):
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pl.DataFrame(
        {
            "REAL": [0, 1, 2],
            "a": [1, 2, 3],
            "b": [0, 2, 0],
            "c": [3, 1, 3],
        }
    )
    default_sheet_df = pl.DataFrame(data, orient="row")
    _create_design_matrix(design_path, design_matrix_df, default_sheet_df)

    with pytest.raises(ValueError, match=error_msg):
        DesignMatrix(design_path, "DesignSheet", "DefaultSheet")


def test_default_values_used(tmp_path):
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pl.DataFrame(
        {
            "REAL": [0, 1, 2, 3],
            "a": [1, 2, 3, 4],
            "b": [0, 2, 0, 1],
            "c": ["low", "high", "medium", "low"],
        }
    )
    default_sheet_df = pl.DataFrame(
        [["one", 1], ["b", 4], ["d", "case_name"]], orient="row"
    )
    _create_design_matrix(design_path, design_matrix_df, default_sheet_df)
    design_matrix = DesignMatrix(design_path, "DesignSheet", "DefaultSheet")
    df = design_matrix.design_matrix_df
    np.testing.assert_equal(df["one"], np.array([1, 1, 1, 1]))
    np.testing.assert_equal(df["b"], np.array([0, 2, 0, 1]))
    np.testing.assert_equal(df["c"], np.array(["low", "high", "medium", "low"]))
    np.testing.assert_equal(
        df["d"],
        np.array(["case_name", "case_name", "case_name", "case_name"]),
    )


def test_whitespace_is_stripped_from_string_parameters(tmp_path):
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pl.DataFrame(
        {
            "REAL": [0, 1, 2, 3],
            "a": [1, 2, 3, 4],
            "b": [0, 2, 0, 1],
            "c": [" low", "high  ", "     medium", "\nlow"],
        }
    )
    default_sheet_df = pl.DataFrame(
        [["one", 1], ["b", 4], ["d", " case_name   "]], orient="row"
    )
    _create_design_matrix(design_path, design_matrix_df, default_sheet_df)
    design_matrix = DesignMatrix(design_path, "DesignSheet", "DefaultSheet")
    df = design_matrix.design_matrix_df
    np.testing.assert_equal(df["one"], np.array([1, 1, 1, 1]))
    np.testing.assert_equal(df["b"], np.array([0, 2, 0, 1]))
    np.testing.assert_equal(df["c"], np.array(["low", "high", "medium", "low"]))
    np.testing.assert_equal(
        df["d"],
        np.array(["case_name", "case_name", "case_name", "case_name"]),
    )
