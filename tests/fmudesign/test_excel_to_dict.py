"""Testing excel_to_dict"""

from datetime import date

import numpy as np
import openpyxl
import pandas as pd
import polars as pl
import pytest

from fmudesign import excel_to_dict, inputdict_to_yaml
from fmudesign._excel_to_dict import (
    _assert_no_merged_cells,
    _has_value,
    _read_dependencies,
)
from fmudesign.utils import map_dependencies

MOCK_GENERAL_INPUT = pd.DataFrame(
    data=[
        ["designtype", "onebyone"],
        ["repeats", "10"],
        ["rms_seeds", "default"],
        ["background", "None"],
        ["distribution_seed", 42],
    ]
)

MOCK_DESIGNINPUT = pd.DataFrame(
    data=[["sensname", "numreal", "type", "param_name"], ["rms_seed", "", "seed"]]
)


def _write_config_workbook(
    path,
    *,
    general_input=MOCK_GENERAL_INPUT,
    design_input=MOCK_DESIGNINPUT,
    defaultvalues=None,
    background=None,
    general_sheet="general_input",
    design_sheet="designinput",
    default_sheet="defaultvalues",
    background_sheet="backgroundsheet",
):
    if defaultvalues is None:
        defaultvalues = pd.DataFrame()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        general_input.to_excel(
            writer, sheet_name=general_sheet, index=False, header=None
        )
        design_input.to_excel(writer, sheet_name=design_sheet, index=False, header=None)
        defaultvalues.to_excel(
            writer, sheet_name=default_sheet, index=False, header=None
        )
        if background is not None:
            background.to_excel(
                writer, sheet_name=background_sheet, index=False, header=None
            )
    return path


def test_that_excel_to_dict_parses_workbook_into_configuration_dictionary(tmp_path):
    input_path = _write_config_workbook(tmp_path / "designinput.xlsx")
    dict_design = excel_to_dict(input_path)

    assert isinstance(dict_design, dict)
    assert dict_design["designtype"] == "onebyone"
    assert dict_design["distribution_seed"] == 42
    assert dict_design["defaultvalues"] == {}
    assert isinstance(dict_design["sensitivities"], dict)

    sens = dict_design["sensitivities"]
    assert sens["rms_seed"]["seedname"] == "RMS_SEED"
    assert sens["rms_seed"]["senstype"] == "seed"

    alternate_path = _write_config_workbook(
        tmp_path / "designinput2.xlsx",
        general_sheet="Generalinput",
        design_sheet="Design_input",
        default_sheet="DefaultValues",
    )
    dict_design = excel_to_dict(alternate_path)
    assert isinstance(dict_design, dict)
    assert dict_design["sensitivities"]["rms_seed"]["senstype"] == "seed"

    yaml_path = tmp_path / "dictdesign.yaml"
    inputdict_to_yaml(dict_design, yaml_path)
    assert "RMS_SEED" in yaml_path.read_text(encoding="utf-8")


def test_that_duplicate_sensitivity_names_raise_value_error(tmp_path):
    mock_erroneous_designinput = pd.DataFrame(
        data=[
            ["sensname", "numreal", "type", "param_name"],
            ["rms_seed", "", "seed"],
            ["rms_seed", "", "seed"],
            [np.nan, "", "seed"],  # NaN sensname - should be ignored
            ["", "", "seed"],  # Empty string - should be ignored
            ["valid_name", "", "seed"],  # Valid unique name
        ]
    )
    input_path = _write_config_workbook(
        tmp_path / "designinput.xlsx", design_input=mock_erroneous_designinput
    )

    with pytest.raises(
        ValueError, match="Two sensitivities cannot share the same sensname"
    ):
        excel_to_dict(input_path)


def test_that_excel_to_dict_strips_sensitivity_and_parameter_name_whitespace(
    tmp_path,
):
    """Spaces before and after parameter names are probably
    invisible user errors in Excel sheets. Remove them.
    """
    mock_spacious_designinput = pd.DataFrame(
        data=[
            ["sensname", "numreal", "type", "param_name"],
            ["rms_seed   ", "", "seed"],
        ]
    )
    defaultvalues_spacious = pd.DataFrame(
        data=[
            ["parametername", "value"],
            ["  spacious_multiplier", 1.2],
            ["spacious2  ", 3.3],
        ]
    )
    input_path = _write_config_workbook(
        tmp_path / "designinput.xlsx",
        design_input=mock_spacious_designinput,
        defaultvalues=defaultvalues_spacious,
    )

    dict_design = excel_to_dict(input_path)
    assert next(iter(dict_design["sensitivities"].keys())) == "rms_seed"
    assert dict_design["defaultvalues"] == {
        "spacious_multiplier": 1.2,
        "spacious2": 3.3,
    }


def test_that_excel_to_dict_rejects_duplicate_default_names_after_trimming(tmp_path):
    input_path = _write_config_workbook(
        tmp_path / "designinput.xlsx",
        defaultvalues=pd.DataFrame(
            [
                ["parametername", "value"],
                ["  a", 1],
                ["a   ", 2],
            ]
        ),
    )

    with pytest.raises(ValueError, match="duplicate parameter names"):
        excel_to_dict(input_path)


def test_that_mixed_sensitivity_types_raise_value_error(tmp_path):
    mock_erroneous_designinput = pd.DataFrame(
        data=[
            ["sensname", "numreal", "type", "param_name"],
            ["rms_seed", "", "seed"],
            ["", "", "dist"],
        ]
    )
    input_path = _write_config_workbook(
        tmp_path / "designinput.xlsx", design_input=mock_erroneous_designinput
    )

    with pytest.raises(ValueError, match="contains more than one sensitivity type"):
        excel_to_dict(input_path)


@pytest.mark.parametrize(
    ("value", "expected"), [(1, True), (np.nan, False), (None, True)]
)
def test_that_has_value_treats_only_nan_as_missing(value, expected):
    assert _has_value(value) is expected


def test_that_excel_to_dict_parses_background_sheet(tmp_path):
    general_input = pd.DataFrame(
        data=[
            ["designtype", "onebyone"],
            ["repeats", 3],
            ["rms_seeds", "default"],
            ["background", "backgroundsheet"],
            ["distribution_seed", 42],
        ]
    )
    defaultvalues = pd.DataFrame(
        data=[["param_name", "default_value"], ["extraseed", "0"]]
    )
    background = pd.DataFrame(
        data=[
            ["param_name", "dist_name", "dist_param1"],
            ["extraseed", "scenario", "30,40,50"],
        ]
    )

    input_path = _write_config_workbook(
        tmp_path / "designinput.xlsx",
        general_input=general_input,
        defaultvalues=defaultvalues,
        background=background,
        design_sheet="design_input",
    )

    dict_design = excel_to_dict(input_path)

    # Assert it has been interpreted correctly from input files:
    assert dict_design["background"]["parameters"]["extraseed"] == [
        "scenario",
        ["30,40,50"],
        None,
    ]
    assert dict_design["repeats"] == 3
    assert dict_design["defaultvalues"]["extraseed"] == 0


def _write_background_workbook(path, background, corr_matrix, background_name):
    general_input = pd.DataFrame(
        data=[
            ["designtype", "onebyone"],
            ["repeats", 3],
            ["rms_seeds", "default"],
            ["background", background_name],
            ["distribution_seed", 42],
        ]
    )
    defaultvalues = pd.DataFrame(
        columns=["param_name", "default_value"], data=[["PARAM_A", 0], ["PARAM_B", 0]]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        general_input.to_excel(
            writer, sheet_name="general_input", index=False, header=None
        )
        MOCK_DESIGNINPUT.to_excel(
            writer, sheet_name="designinput", index=False, header=None
        )
        defaultvalues.to_excel(writer, sheet_name="defaultvalues", index=False)
        background.to_excel(
            writer, sheet_name="backgroundsheet", index=False, header=None
        )
        if corr_matrix is not None:
            corr_matrix.to_excel(writer, sheet_name="bgcorr")
    return path


@pytest.mark.parametrize("sheet", ["designinput", "backgroundsheet"])
@pytest.mark.parametrize(
    ("missing_field", "error"),
    [
        (
            "param_name",
            (
                r"(Dist sensitivity uncertainty|Background parameters)"
                r".*empty parameter name"
            ),
        ),
        (
            "dist_param1",
            r"Parameter PARAM_A .*empty first distribution parameter",
        ),
        (
            "dist_param2",
            r'Parameter PARAM_A .*"dist_param3" while "dist_param2" is empty',
        ),
        (
            "dist_param3",
            r'Parameter PARAM_A .*"dist_param4" while "dist_param3" is empty',
        ),
    ],
    ids=[
        "missing-name",
        "missing-first-argument",
        "gap-before-third",
        "gap-before-fourth",
    ],
)
def test_that_missing_distribution_fields_report_the_field_and_source(
    tmp_path, sheet, missing_field, error
):
    row = {
        "param_name": "PARAM_A",
        "dist_name": "normal",
        "dist_param1": 0,
        "dist_param2": 1,
        "dist_param3": -1,
        "dist_param4": 1,
    }
    row[missing_field] = None
    if sheet == "designinput":
        row = {"sensname": "uncertainty", "type": "dist", **row}
    distribution_rows = pd.DataFrame([list(row), list(row.values())])
    input_path = tmp_path / "designinput.xlsx"

    if sheet == "designinput":
        _write_config_workbook(input_path, design_input=distribution_rows)
    else:
        _write_background_workbook(input_path, distribution_rows, None, sheet)

    with pytest.raises(ValueError, match=error) as exc_info:
        excel_to_dict(input_path)

    message = str(exc_info.value).lower()
    if sheet == "backgroundsheet":
        assert "background" in message
    else:
        assert "background" not in message


BACKGROUND_WITH_CORR = pd.DataFrame(
    data=[
        ["param_name", "dist_name", "dist_param1", "dist_param2", "corr_sheet"],
        ["PARAM_A", "uniform", 0, 1, "bgcorr"],
        ["PARAM_B", "uniform", 0, 1, "bgcorr"],
    ]
)


@pytest.mark.parametrize(
    ("index", "columns", "error"),
    [
        (
            ["PARAM_A", "PARAM_B"],
            ["PARAM_A", "PARAM_TYPO"],
            "Mismatch between column and index in correlation",
        ),
        (
            ["PARAM_A", "PARAM_TYPO"],
            ["PARAM_A", "PARAM_TYPO"],
            "Mismatch between parameters",
        ),
    ],
)
def test_that_invalid_background_correlation_sheet_raises_value_error(
    tmp_path, index, columns, error
):
    corr_matrix = pd.DataFrame(
        [[1.0, np.nan], [0.5, 1.0]],
        index=index,
        columns=columns,
    )
    input_path = _write_background_workbook(
        tmp_path / "designinput.xlsx",
        BACKGROUND_WITH_CORR,
        corr_matrix,
        "backgroundsheet",
    )

    with pytest.raises(ValueError, match=error):
        excel_to_dict(input_path)


def test_that_missing_background_sheet_error_lists_available_sheets(tmp_path):
    input_path = _write_background_workbook(
        tmp_path / "designinput.xlsx", BACKGROUND_WITH_CORR, None, "typo_sheet"
    )

    with pytest.raises(ValueError, match="Sheets in workbook") as exc_info:
        excel_to_dict(input_path)

    message = str(exc_info.value)
    assert "typo_sheet" in message
    assert "backgroundsheet" in message
    assert "Use 'None' as background" in message


@pytest.mark.parametrize(
    "background_name", ["Backgroundsheet", "background_sheet", " backgroundsheet "]
)
def test_that_background_sheet_matching_ignores_case_underscores_and_whitespace(
    tmp_path, background_name
):
    corr_matrix = pd.DataFrame(
        [[1.0, np.nan], [0.5, 1.0]],
        index=["PARAM_A", "PARAM_B"],
        columns=["PARAM_A", "PARAM_B"],
    )
    input_path = _write_background_workbook(
        tmp_path / "designinput.xlsx",
        BACKGROUND_WITH_CORR,
        corr_matrix,
        background_name,
    )

    background = excel_to_dict(input_path)["background"]
    assert list(background["parameters"]) == ["PARAM_A", "PARAM_B"]
    assert background["correlations"]["sheetnames"] == ["bgcorr"]


def test_that_missing_background_csv_file_raises_value_error(use_tmpdir):
    input_path = _write_background_workbook(
        "designinput.xlsx",
        BACKGROUND_WITH_CORR,
        None,
        "missing_background.csv",
    )

    with pytest.raises(
        ValueError,
        match=r"Sheet 'missing_background.csv' with background parameters, "
        "specified in the general input sheet, "
        "was not found in 'designinput.xlsx'.",
    ):
        excel_to_dict(input_path)


@pytest.mark.parametrize("background_name", ["None", "none", np.nan])
def test_that_none_like_background_names_disable_background(tmp_path, background_name):
    input_path = _write_background_workbook(
        tmp_path / "designinput.xlsx", BACKGROUND_WITH_CORR, None, background_name
    )

    assert excel_to_dict(input_path)["background"] is None


def test_that_assert_no_merged_cells_rejects_merged_cells(tmp_path):
    input_path = tmp_path / "test_file.xlsx"
    test_data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    test_data.to_excel(input_path, sheet_name="sheet1", index=False)

    workbook = openpyxl.load_workbook(input_path)
    workbook["sheet1"].merge_cells("A1:B1")
    workbook.save(input_path)
    workbook.close()

    with pytest.raises(ValueError, match="Merged cells"):
        _assert_no_merged_cells(input_path)


def test_that_excel_to_dict_preserves_seed_strategy(tmp_path):
    general = pd.DataFrame(
        data=[
            ["designtype", "onebyone"],
            ["repeats", "10"],
            ["rms_seeds", "default"],
            ["background", "None"],
            ["distribution_seed", 42],
            ["seed_strategy", "independent"],
        ]
    )
    designinput = pd.DataFrame(
        data=[["sensname", "numreal", "type", "param_name"], ["rms_seed", "", "seed"]]
    )
    input_path = _write_config_workbook(
        tmp_path / "designinput.xlsx",
        general_input=general,
        design_input=designinput,
    )
    dict_design = excel_to_dict(input_path)
    assert dict_design["seed_strategy"] == "independent"


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
        pl.DataFrame([pl.Series("SOURCE", ["low", 2, " high "], dtype=pl.Object)]),
        dependencies={"SOURCE": dependencies},
    )

    assert result.to_dict(as_series=False) == {
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
        pl.DataFrame(
            [pl.Series("SOURCE", [2, "other", 1.9999999999], dtype=pl.Object)]
        ),
        dependencies={"SOURCE": dependencies},
    )

    assert result.to_dict(as_series=False) == {
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
        pl.DataFrame({"SOURCE": ["2018-11-02 00:00:00", "2018-11-02"]}),
        dependencies={"SOURCE": dependencies},
    )

    assert result.to_dict(as_series=False) == {
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
        pl.DataFrame({"SOURCE": ["True", "true", "TRUE"]}),
        dependencies={"SOURCE": dependencies},
    )

    assert result.to_dict(as_series=False) == {
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
        ValueError,
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
        pl.DataFrame({"SOURCE": ["C1", "C2"]}),
        dependencies={"SOURCE": dependencies},
    )

    assert result.to_dict(as_series=False) == {
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
        ValueError, match="Duplicate parameter names in dependency sheet 'dependencies'"
    ):
        _read_dependencies(
            filename=str(input_path), sheetname="dependencies", from_parameter="SOURCE"
        )


def test_that_missing_dependency_worksheet_raises_value_error(tmp_path):
    input_path = _write_dependency_workbook(
        tmp_path / "dependencies.xlsx", [["SOURCE", "TARGET"]]
    )

    with pytest.raises(ValueError, match="Worksheet 'missing' not found"):
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
        ValueError,
        match=r"Parameter SOURCE.*sheet specifying the dependencies dependencies.*"
        r"does not contain the input parameter",
    ):
        _read_dependencies(
            filename=str(input_path),
            sheetname="dependencies",
            from_parameter="SOURCE",
        )
