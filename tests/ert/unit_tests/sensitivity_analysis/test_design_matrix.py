import numpy as np
import pandas as pd
import pytest

from ert.config.design_matrix import DESIGN_MATRIX_GROUP, DesignMatrix
from ert.config.gen_kw_config import GenKwConfig, TransformFunctionDefinition


def _create_design_matrix(xls_path, design_matrix_df, default_sheet_df) -> DesignMatrix:
    with pd.ExcelWriter(xls_path) as xl_write:
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="DesignSheet01")
        default_sheet_df.to_excel(
            xl_write, index=False, sheet_name="DefaultValues", header=False
        )
    return DesignMatrix(xls_path, "DesignSheet01", "DefaultValues")


@pytest.mark.parametrize(
    "design_sheet_pd, default_sheet_pd, error_msg",
    [
        pytest.param(
            pd.DataFrame(
                {
                    "REAL": [0, 1, 2],
                    "c": [1, 2, 3],
                    "d": [0, 2, 0],
                }
            ),
            pd.DataFrame([["e", 1]]),
            "",
            id="ok_merge",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "REAL": [0, 1, 2],
                    "a": [1, 2, 3],
                }
            ),
            pd.DataFrame([["e", 1]]),
            "Design Matrices do not have unique keys",
            id="not_unique_keys",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "REAL": [0, 1],
                    "d": [1, 2],
                }
            ),
            pd.DataFrame([["e", 1]]),
            "Design Matrices don't have the same active realizations!",
            id="not_same_acitve_realizations",
        ),
    ],
)
def test_merge_multiple_occurrences(
    tmp_path, design_sheet_pd, default_sheet_pd, error_msg
):
    design_matrix_1 = _create_design_matrix(
        tmp_path / "design_matrix_1.xlsx",
        pd.DataFrame(
            {
                "REAL": [0, 1, 2],
                "a": [1, 2, 3],
                "b": [0, 2, 0],
            },
        ),
        pd.DataFrame([["a", 1], ["b", 4]]),
    )

    design_matrix_2 = _create_design_matrix(
        tmp_path / "design_matrix_2.xlsx", design_sheet_pd, default_sheet_pd
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
        np.testing.assert_equal(df["c"], np.array([1, 2, 3]))
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
            "Multiple overlapping groups with design matrix found in existing parameters!",
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
                    template_file="",
                    transform_function_definitions=[
                        TransformFunctionDefinition(param, "UNIFORM", [0, 1])
                        for param in parameters[group_name]
                    ],
                    output_file="kw.txt",
                    update=True,
                )
            )

    realizations = [0, 1, 2]
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pd.DataFrame(
        {
            "REAL": realizations,
            "a": [1, 2, 3],
            "b": [0, 2, 0],
        }
    )
    default_sheet_df = pd.DataFrame([["a", 1], ["b", 4]])
    with pd.ExcelWriter(design_path) as xl_write:
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="DesignSheet01")
        default_sheet_df.to_excel(
            xl_write, index=False, sheet_name="DefaultValues", header=False
        )
    design_matrix = DesignMatrix(design_path, "DesignSheet01", "DefaultValues")
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
    design_matrix_df = pd.DataFrame(
        {
            "REAL": realizations,
            "a": [1, 2, 3],
            "b": [0, 2, 0],
            "c": ["low", "high", "medium"],
        }
    )
    default_sheet_df = pd.DataFrame([["one", 1], ["b", 4], ["d", "case_name"]])
    with pd.ExcelWriter(design_path) as xl_write:
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="DesignSheet01")
        default_sheet_df.to_excel(
            xl_write, index=False, sheet_name="DefaultValues", header=False
        )
    design_matrix = DesignMatrix(design_path, "DesignSheet01", "DefaultValues")
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
    design_matrix_df = pd.DataFrame(
        {
            "REAL": real_column,
            "a": [1, 2, 3],
            "b": [0, 2, 0],
            "c": [3, 1, 3],
        }
    )
    default_sheet_df = pd.DataFrame()
    with pd.ExcelWriter(design_path) as xl_write:
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="DesignSheet01")
        default_sheet_df.to_excel(
            xl_write, index=False, sheet_name="DefaultValues", header=False
        )

    with pytest.raises(ValueError, match=error_msg):
        DesignMatrix(design_path, "DesignSheet01", "DefaultValues")


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
            ["a", "b  ", ""],
            r"Empty parameter name found in column 2",
            id="missing entries",
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
            r"Multiple words in parameter name found in column 1 \(b c d e\)\.\nNumeric parameter name found in column 2",
            id="multiple errors",
        ),
    ],
)
def test_reading_design_matrix_validate_headers(tmp_path, column_names, error_msg):
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pd.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=column_names
    )
    default_sheet_df = pd.DataFrame([["one", 1], ["b", 4], ["d", 6]])
    with pd.ExcelWriter(design_path) as xl_write:
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="DesignSheet01")
        default_sheet_df.to_excel(
            xl_write, index=False, sheet_name="DefaultValues", header=False
        )

    with pytest.raises(ValueError, match=error_msg):
        DesignMatrix(design_path, "DesignSheet01", "DefaultValues")


@pytest.mark.parametrize(
    "values, error_msg",
    [
        pytest.param(
            [0, pd.NA, 1],
            r"Design matrix contains empty cells \['Row 1, column 1'\]",
            id="duplicate entries",
        ),
        pytest.param(
            [0, "some", np.nan],
            r"Design matrix contains empty cells \['Row 2, column 1'\]",
            id="invalid float values",
        ),
    ],
)
def test_reading_design_matrix_validate_cells(tmp_path, values, error_msg):
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pd.DataFrame(
        {
            "REAL": [1, 5, 7],
            "a": values,
            "b": [0, 2, 0],
            "c": [3, 1, 3],
        }
    )
    default_sheet_df = pd.DataFrame()
    with pd.ExcelWriter(design_path) as xl_write:
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="DesignSheet01")
        default_sheet_df.to_excel(
            xl_write, index=False, sheet_name="DefaultValues", header=False
        )

    with pytest.raises(ValueError, match=error_msg):
        DesignMatrix(design_path, "DesignSheet01", "DefaultValues")


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
            [[2, 1], ["b", ""], ["d", 6]],
            r"Default sheet contains empty cells \['Row 1, column 1'\]",
            id="numerical entries as param names",
        ),
        pytest.param(
            [[" a", 1], ["a ", "some"], ["d", 6]],
            r"Default sheet contains duplicate parameter names",
            id="duplicate parameter names",
        ),
    ],
)
def test_reading_default_sheet_validation(tmp_path, data, error_msg):
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pd.DataFrame(
        {
            "REAL": [0, 1, 2],
            "a": [1, 2, 3],
            "b": [0, 2, 0],
            "c": [3, 1, 3],
        }
    )
    default_sheet_df = pd.DataFrame(data)
    with pd.ExcelWriter(design_path) as xl_write:
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="DesignSheet01")
        default_sheet_df.to_excel(
            xl_write, index=False, sheet_name="DefaultValues", header=False
        )

    with pytest.raises(ValueError, match=error_msg):
        DesignMatrix(design_path, "DesignSheet01", "DefaultValues")


def test_default_values_used(tmp_path):
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pd.DataFrame(
        {
            "REAL": [0, 1, 2, 3],
            "a": [1, 2, 3, 4],
            "b": [0, 2, 0, 1],
            "c": ["low", "high", "medium", "low"],
        }
    )
    default_sheet_df = pd.DataFrame([["one", 1], ["b", 4], ["d", "case_name"]])
    with pd.ExcelWriter(design_path) as xl_write:
        design_matrix_df.to_excel(xl_write, index=False, sheet_name="DesignSheet01")
        default_sheet_df.to_excel(
            xl_write, index=False, sheet_name="DefaultValues", header=False
        )
    design_matrix = DesignMatrix(design_path, "DesignSheet01", "DefaultValues")
    df = design_matrix.design_matrix_df
    np.testing.assert_equal(df["one"], np.array([1, 1, 1, 1]))
    np.testing.assert_equal(df["b"], np.array([0, 2, 0, 1]))
    np.testing.assert_equal(
        df["d"],
        np.array(["case_name", "case_name", "case_name", "case_name"]),
    )
