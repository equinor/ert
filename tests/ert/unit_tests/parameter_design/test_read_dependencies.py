from datetime import date

import numpy as np
import openpyxl
import pandas as pd
import pytest

from ert.parameter_design import FmudesignError
from ert.parameter_design.design_input import _read_dependencies
from ert.parameter_design.utils import _has_value, map_dependencies


@pytest.mark.parametrize(
    ("value", "expected"), [(1, True), (np.nan, False), (None, True)]
)
def test_that_has_value_treats_only_nan_as_missing(value, expected):
    assert _has_value(value) is expected


def _write_dependency_workbook(path, rows):
    workbook = openpyxl.Workbook()
    sheet = workbook.create_sheet("dependencies")
    for row in rows:
        sheet.append(row)
    workbook.save(path)
    workbook.close()
    return path


def test_that_dependency_columns_map_by_parameter_name_and_exact_source_value(tmp_path):
    input_path = _write_dependency_workbook(
        tmp_path / "dependencies.xlsx",
        [
            ["MULTIPLIER", "SOURCE", "LABEL"],
            [1, " high ", " sand "],
            [2.5, 2, "mixed"],
            [-3, "low", "shale"],
        ],
    )
    dependencies = _read_dependencies(
        filename=str(input_path), sheetname="dependencies", from_parameter="SOURCE"
    )

    result = map_dependencies(
        pd.DataFrame({"SOURCE": ["low", 2, " high "]}),
        dependencies={"SOURCE": dependencies},
    )

    assert result.to_dict(orient="list") == {
        "SOURCE": ["low", 2, " high "],
        "MULTIPLIER": [-3, 2.5, 1],
        "LABEL": ["shale", "mixed", " sand "],
    }


def test_that_numeric_dependency_keys_and_targets_preserve_precision(tmp_path):
    input_path = _write_dependency_workbook(
        tmp_path / "dependencies.xlsx",
        [
            ["SOURCE", "TARGET"],
            [1.9999999999, 1e-10],
            [2, 0.123456789012345],
            ["other", -1e-10],
        ],
    )
    dependencies = _read_dependencies(
        filename=str(input_path), sheetname="dependencies", from_parameter="SOURCE"
    )

    result = map_dependencies(
        pd.DataFrame({"SOURCE": [2, "other", 1.9999999999]}),
        dependencies={"SOURCE": dependencies},
    )

    assert result.to_dict(orient="list") == {
        "SOURCE": [2, "other", 1.9999999999],
        "TARGET": [0.123456789012345, -1e-10, 1e-10],
    }


def test_that_native_excel_dates_and_text_timestamps_map_to_distinct_values(tmp_path):
    input_path = _write_dependency_workbook(
        tmp_path / "dependencies.xlsx",
        [
            ["SOURCE", "TARGET"],
            [date(2018, 11, 2), "native-date"],
            ["2018-11-02 00:00:00", "text-timestamp"],
        ],
    )
    dependencies = _read_dependencies(
        filename=str(input_path), sheetname="dependencies", from_parameter="SOURCE"
    )

    result = map_dependencies(
        pd.DataFrame({"SOURCE": ["2018-11-02 00:00:00", "2018-11-02"]}),
        dependencies={"SOURCE": dependencies},
    )

    assert result.to_dict(orient="list") == {
        "SOURCE": ["2018-11-02 00:00:00", "2018-11-02"],
        "TARGET": ["text-timestamp", "native-date"],
    }


def test_that_excel_booleans_match_title_case_dependency_categories(tmp_path):
    input_path = _write_dependency_workbook(
        tmp_path / "dependencies.xlsx",
        [
            ["SOURCE", "TARGET"],
            [True, False],
            ["true", "false"],
            ["TRUE", "FALSE"],
        ],
    )
    dependencies = _read_dependencies(
        filename=str(input_path), sheetname="dependencies", from_parameter="SOURCE"
    )

    result = map_dependencies(
        pd.DataFrame({"SOURCE": ["True", "true", "TRUE"]}),
        dependencies={"SOURCE": dependencies},
    )

    assert result.to_dict(orient="list") == {
        "SOURCE": ["True", "true", "TRUE"],
        "TARGET": ["False", "false", "FALSE"],
    }


def test_that_unnamed_dependency_columns_and_blank_rows_are_ignored(tmp_path):
    input_path = _write_dependency_workbook(
        tmp_path / "dependencies.xlsx",
        [
            ["SOURCE", None, "TARGET", "", " \t "],
            ["high", "note", "upper", "note", "note"],
            [None, None, None, None, None],
            [None, "note", None, "note", "note"],
            [" \t", "note", "\n ", "note", "note"],
            ["low", "note", "lower", "note", "note"],
        ],
    )

    assert _read_dependencies(
        filename=str(input_path), sheetname="dependencies", from_parameter="SOURCE"
    ) == {
        "from_values": ["high", "low"],
        "to_params": {"TARGET": ["upper", "lower"]},
    }


@pytest.mark.parametrize(
    ("row", "missing_parameter"),
    [
        pytest.param([None, 0], "SOURCE", id="missing-source"),
        pytest.param([" \t", 0], "SOURCE", id="whitespace-source"),
        pytest.param(["middle", None], "TARGET", id="missing-target"),
        pytest.param(["middle", ""], "TARGET", id="empty-target"),
        pytest.param(["middle", " \t"], "TARGET", id="whitespace-target"),
        pytest.param(["middle", "#N/A"], "TARGET", id="excel-error"),
    ],
)
def test_that_incomplete_dependency_rows_raise_a_configuration_error(
    tmp_path, row, missing_parameter
):
    input_path = _write_dependency_workbook(
        tmp_path / "dependencies.xlsx",
        [["SOURCE", "TARGET"], ["high", 20], [None, None], row],
    )

    with pytest.raises(
        FmudesignError,
        match=f"Missing dependency value for parameter '{missing_parameter}' "
        "in sheet 'dependencies', row 4",
    ):
        _read_dependencies(
            filename=str(input_path), sheetname="dependencies", from_parameter="SOURCE"
        )


def test_that_dependency_headers_without_rows_copy_source_values(tmp_path):
    input_path = _write_dependency_workbook(
        tmp_path / "dependencies.xlsx", [["SOURCE", None, "COPY", ""]]
    )
    dependencies = _read_dependencies(
        filename=str(input_path), sheetname="dependencies", from_parameter="SOURCE"
    )

    result = map_dependencies(
        pd.DataFrame({"SOURCE": ["C1", "C2"]}),
        dependencies={"SOURCE": dependencies},
    )

    assert result.to_dict(orient="list") == {
        "SOURCE": ["C1", "C2"],
        "COPY": ["C1", "C2"],
    }


@pytest.mark.parametrize("duplicate_parameter", ["SOURCE", "TARGET"])
def test_that_duplicate_dependency_parameter_names_raise_value_error(
    tmp_path, duplicate_parameter
):
    input_path = _write_dependency_workbook(
        tmp_path / "dependencies.xlsx",
        [["SOURCE", "TARGET", duplicate_parameter]],
    )

    with pytest.raises(
        FmudesignError,
        match="Duplicate parameter names in dependency sheet 'dependencies'",
    ):
        _read_dependencies(
            filename=str(input_path), sheetname="dependencies", from_parameter="SOURCE"
        )


def test_that_missing_dependency_worksheet_raises_value_error(tmp_path):
    input_path = _write_dependency_workbook(
        tmp_path / "dependencies.xlsx", [["SOURCE", "TARGET"]]
    )

    with pytest.raises(FmudesignError, match="Worksheet 'missing' not found"):
        _read_dependencies(
            filename=str(input_path), sheetname="missing", from_parameter="SOURCE"
        )


@pytest.mark.parametrize(
    "rows", [[], [["OTHER"], [1]]], ids=["empty-sheet", "missing-source"]
)
def test_that_dependency_sheet_without_source_parameter_raises_value_error(
    tmp_path, rows
):
    input_path = _write_dependency_workbook(tmp_path / "dependencies.xlsx", rows)

    with pytest.raises(
        FmudesignError,
        match=r"Parameter SOURCE.*sheet specifying the dependencies dependencies.*"
        r"does not contain the input parameter",
    ):
        _read_dependencies(
            filename=str(input_path),
            sheetname="dependencies",
            from_parameter="SOURCE",
        )
