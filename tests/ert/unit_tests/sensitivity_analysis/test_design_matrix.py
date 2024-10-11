import numpy as np
import pandas as pd
import pytest

from ert.config.design_matrix import DESIGN_MATRIX_GROUP, DesignMatrix


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
    design_matrix.read_design_matrix()
    design_params = design_matrix.parameter_configuration.get(DESIGN_MATRIX_GROUP, [])
    assert all(param in design_params for param in ("a", "b", "c", "one", "d"))
    assert design_matrix.num_realizations == 3
    assert design_matrix.active_realizations == [True, True, False, False, True]


@pytest.mark.parametrize(
    "real_column, error_msg",
    [
        pytest.param([0, 1, 1], "Index has duplicate keys", id="duplicate entries"),
        pytest.param(
            [0, 1.1, 2],
            "REAL column must only contain positive integers",
            id="invalid float values",
        ),
        pytest.param(
            [0, "a", 10],
            "REAL column must only contain positive integers",
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
    design_matrix = DesignMatrix(design_path, "DesignSheet01", "DefaultValues")
    with pytest.raises(ValueError, match=error_msg):
        design_matrix.read_design_matrix()


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
            r"Column headers not present in column \[2\]",
            id="missing entries",
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
    design_matrix = DesignMatrix(design_path, "DesignSheet01", "DefaultValues")
    with pytest.raises(ValueError, match=error_msg):
        design_matrix.read_design_matrix()


@pytest.mark.parametrize(
    "values, error_msg",
    [
        pytest.param(
            [0, pd.NA, 1],
            r"Design matrix contains empty cells \['Realization 5, column a'\]",
            id="duplicate entries",
        ),
        pytest.param(
            [0, "some", np.nan],
            r"Design matrix contains empty cells \['Realization 7, column a'\]",
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
    design_matrix = DesignMatrix(design_path, "DesignSheet01", "DefaultValues")
    with pytest.raises(ValueError, match=error_msg):
        design_matrix.read_design_matrix()


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
    design_matrix = DesignMatrix(design_path, "DesignSheet01", "DefaultValues")
    with pytest.raises(ValueError, match=error_msg):
        design_matrix.read_design_matrix()


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
    design_matrix.read_design_matrix()
    df = design_matrix.design_matrix_df
    np.testing.assert_equal(df[DESIGN_MATRIX_GROUP, "one"], np.array([1, 1, 1, 1]))
    np.testing.assert_equal(df[DESIGN_MATRIX_GROUP, "b"], np.array([0, 2, 0, 1]))
    np.testing.assert_equal(
        df[DESIGN_MATRIX_GROUP, "d"],
        np.array(["case_name", "case_name", "case_name", "case_name"]),
    )
