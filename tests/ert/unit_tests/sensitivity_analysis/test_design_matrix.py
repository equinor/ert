import numpy as np
import polars as pl
import pytest
from xlsxwriter import Workbook

from ert.config import DataSource, DesignMatrix, GenKwConfig
from ert.config.design_matrix import DESIGN_MATRIX_GROUP
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
        tmp_path / "design_matrix_1.xlsx", "DesignSheet", "DefaultSheet", "sampled"
    )
    assert design_matrix_1.priority_source == "sampled"
    assert design_matrix_1.parameter_priority == {
        "a": DataSource.SAMPLED,
        "b": DataSource.SAMPLED,
    }
    _create_design_matrix(
        tmp_path / "design_matrix_2.xlsx", design_sheet_pd, default_sheet_pd
    )
    design_matrix_2 = DesignMatrix(
        tmp_path / "design_matrix_2.xlsx",
        "DesignSheet",
        "DefaultSheet",
        "design_matrix",
    )
    assert design_matrix_2.priority_source == "design_matrix"
    assert set(design_matrix_2.parameter_priority.values()) == {
        DataSource.DESIGN_MATRIX
    }

    if error_msg:
        with pytest.raises(ValueError, match=error_msg):
            design_matrix_1.merge_with_other(design_matrix_2)
    else:
        design_matrix_1.merge_with_other(design_matrix_2)
        design_params = [cfg.name for cfg in design_matrix_1.parameter_configurations]
        assert all(param in design_params for param in ("a", "b", "c", "d"))
        assert design_matrix_1.active_realizations == [True, True, True]
        df = design_matrix_1.design_matrix_df
        np.testing.assert_equal(df["a"], np.array([1, 2, 3]))
        np.testing.assert_equal(df["b"], np.array([0, 2, 0]))
        np.testing.assert_equal(df["c"], np.array([9, 10, 11.1]))
        np.testing.assert_equal(df["d"], np.array([0, 2, 0]))

        expected_priority = {
            "a": (
                DataSource.DESIGN_MATRIX
                if "a" in design_sheet_pd.columns
                else DataSource.SAMPLED
            ),
            "b": DataSource.SAMPLED,
            "c": DataSource.DESIGN_MATRIX,
            "d": DataSource.DESIGN_MATRIX,
            "e": DataSource.DESIGN_MATRIX,
        }
        assert design_matrix_1.parameter_priority == expected_priority


@pytest.mark.parametrize(
    "parameters, priority, num_configs, input_source, group_name",
    [
        pytest.param(
            ["a", "b"],
            "design_matrix",
            2,
            {"a": DataSource.DESIGN_MATRIX, "b": DataSource.DESIGN_MATRIX},
            {"a": DESIGN_MATRIX_GROUP, "b": DESIGN_MATRIX_GROUP},
            id="overlap_priority_design_matrix",
        ),
        pytest.param(
            ["a", "b"],
            "sampled",
            2,
            {"a": DataSource.SAMPLED, "b": DataSource.SAMPLED},
            {"a": "COEFFS", "b": "COEFFS"},
            id="overlap_priority_sampled",
        ),
    ],
)
def test_merge_with_existing_parameters_with_custom_priorities(
    tmp_path, priority, parameters, num_configs, input_source, group_name
):
    genkw_configs = [
        GenKwConfig(
            name=param,
            group="COEFFS",
            distribution={"name": "uniform", "min": 0, "max": 1},
        )
        for param in parameters
    ]

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
    design_matrix = DesignMatrix(design_path, "DesignSheet", "DefaultSheet", priority)
    new_config_parameters = design_matrix.merge_with_existing_parameters(genkw_configs)
    assert len(new_config_parameters) == num_configs
    for config in new_config_parameters:
        assert config.name in input_source
        assert config.input_source == input_source[config.name], (
            f"{config} mismatch in input source"
        )
        assert config.group == group_name[config.name], (
            f"{config} mismatch in group name"
        )


@pytest.mark.parametrize(
    "parameters, num_configs, priority_source, res_input_source, res_group_name,",
    [
        pytest.param(
            ["a", "b"],
            2,
            "design_matrix",
            {"a": DataSource.DESIGN_MATRIX, "b": DataSource.DESIGN_MATRIX},
            {"a": DESIGN_MATRIX_GROUP, "b": DESIGN_MATRIX_GROUP},
            id="genkw_overlap_design_matrix",
        ),
        pytest.param(
            ["a", "b"],
            2,
            "sampled",
            {"a": DataSource.SAMPLED, "b": DataSource.SAMPLED},
            {"a": "COEFFS", "b": "COEFFS"},
            id="genkw_overlap_sampled",
        ),
        pytest.param(
            ["aa", "bb"],
            4,
            "design_matrix",
            {
                "a": DataSource.DESIGN_MATRIX,
                "b": DataSource.DESIGN_MATRIX,
                "aa": DataSource.SAMPLED,
                "bb": DataSource.SAMPLED,
            },
            {
                "a": DESIGN_MATRIX_GROUP,
                "b": DESIGN_MATRIX_GROUP,
                "aa": "COEFFS",
                "bb": "COEFFS",
            },
            id="genkw_added",
        ),
        pytest.param(
            ["a", "bb"],
            3,
            "design_matrix",
            {
                "a": DataSource.DESIGN_MATRIX,
                "b": DataSource.DESIGN_MATRIX,
                "bb": DataSource.SAMPLED,
            },
            {
                "a": DESIGN_MATRIX_GROUP,
                "b": DESIGN_MATRIX_GROUP,
                "bb": "COEFFS",
            },
            id="genkw_added_and_overlap",
        ),
    ],
)
def test_read_and_merge_with_existing_parameters(
    tmp_path, parameters, num_configs, priority_source, res_input_source, res_group_name
):
    genkw_configs = [
        GenKwConfig(
            name=param,
            group="COEFFS",
            distribution={"name": "uniform", "min": 0, "max": 1},
        )
        for param in parameters
    ]

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
    design_matrix = DesignMatrix(
        design_path, "DesignSheet", "DefaultSheet", priority_source
    )
    new_config_parameters = design_matrix.merge_with_existing_parameters(genkw_configs)
    assert len(new_config_parameters) == num_configs
    for config in new_config_parameters:
        assert config.name in res_input_source
        assert config.input_source == res_input_source[config.name], (
            f"{config} mismatch in input source"
        )
        assert config.group == res_group_name[config.name], (
            f"{config} mismatch in group name"
        )


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
    design_params = [cfg.name for cfg in design_matrix.parameter_configurations]
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


def test_that_numeric_string_columns_are_converted(tmp_path):
    realizations = [0, 1, 4]
    design_path = tmp_path / "design_matrix.xlsx"
    design_matrix_df = pl.DataFrame(
        {
            "REAL": realizations,
            "a": ["1", "2", "3"],
            "b": [0, "2.2", "0.1"],
            "c": [10, "high", "medium"],
        },
        strict=False,
    )
    _create_design_matrix(design_path, design_matrix_df)
    design_matrix = DesignMatrix(design_path, "DesignSheet", None)
    df = design_matrix.design_matrix_df
    assert df.schema["a"] == pl.Int64
    assert df.schema["b"] == pl.Float64
    assert df.schema["c"] == pl.String
    np.testing.assert_equal(df["a"], np.array([1, 2, 3]))
    np.testing.assert_equal(df["b"], np.array([0, 2.2, 0.1]))
    np.testing.assert_equal(df["c"], np.array(["10", "high", "medium"]))
