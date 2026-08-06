"""Testing excel_to_dict"""

from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest

from semeio.fmudesign import excel_to_dict, inputdict_to_yaml
from semeio.fmudesign._excel_to_dict import _assert_no_merged_cells, _has_value

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


def test_excel_to_dict(tmpdir, monkeypatch):
    """Test that we can convert an Excelfile to a dictionary"""
    monkeypatch.chdir(tmpdir)
    defaultvalues = pd.DataFrame()
    # pylint: disable=abstract-class-instantiated
    writer = pd.ExcelWriter("designinput.xlsx", engine="openpyxl")
    MOCK_GENERAL_INPUT.to_excel(
        writer, sheet_name="general_input", index=False, header=None
    )
    MOCK_DESIGNINPUT.to_excel(
        writer, sheet_name="designinput", index=False, header=None
    )
    defaultvalues.to_excel(writer, sheet_name="defaultvalues", index=False, header=None)
    writer.close()

    dict_design = excel_to_dict("designinput.xlsx")
    assert isinstance(dict_design, dict)
    assert dict_design["designtype"] == "onebyone"
    assert dict_design["distribution_seed"] == 42
    assert "defaultvalues" in dict_design

    assert isinstance(dict_design["defaultvalues"], dict)
    assert not dict_design["defaultvalues"]  # (it is empty)

    assert isinstance(dict_design["sensitivities"], dict)

    sens = dict_design["sensitivities"]
    # This now contains a key for each sensitivity to make
    assert "rms_seed" in sens
    assert isinstance(sens["rms_seed"], dict)
    assert sens["rms_seed"]["seedname"] == "RMS_SEED"
    assert sens["rms_seed"]["senstype"] == "seed"  # upper-cased

    # Check that we can vary some strings
    writer = pd.ExcelWriter("designinput2.xlsx", engine="openpyxl")
    MOCK_GENERAL_INPUT.to_excel(
        writer, sheet_name="Generalinput", index=False, header=None
    )
    MOCK_DESIGNINPUT.to_excel(
        writer, sheet_name="Design_input", index=False, header=None
    )
    defaultvalues.to_excel(writer, sheet_name="DefaultValues", index=False, header=None)
    writer.close()

    dict_design = excel_to_dict("designinput2.xlsx")
    assert isinstance(dict_design, dict)
    assert dict_design["sensitivities"]["rms_seed"]["senstype"] == "seed"

    # Dump to yaml:
    inputdict_to_yaml(dict_design, "dictdesign.yaml")
    assert Path("dictdesign.yaml").exists()
    assert "RMS_SEED" in Path("dictdesign.yaml").read_text(encoding="utf-8")


def test_duplicate_sensname_exception(tmpdir, monkeypatch):
    """Test that exceptions are raised for erroneous sensnames"""
    # pylint: disable=abstract-class-instantiated
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
    monkeypatch.chdir(tmpdir)
    defaultvalues = pd.DataFrame()

    writer = pd.ExcelWriter("designinput3.xlsx", engine="openpyxl")
    MOCK_GENERAL_INPUT.to_excel(
        writer, sheet_name="general_input", index=False, header=None
    )
    mock_erroneous_designinput.to_excel(
        writer, sheet_name="designinput", index=False, header=None
    )
    defaultvalues.to_excel(writer, sheet_name="defaultvalues", index=False, header=None)
    writer.close()

    with pytest.raises(
        ValueError, match="Two sensitivities can not share the same sensname"
    ):
        excel_to_dict("designinput3.xlsx")


def test_strip_spaces(tmpdir, monkeypatch):
    """Spaces before and after parameter names are probably
    invisible user errors in Excel sheets. Remove them."""
    # pylint: disable=abstract-class-instantiated
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
    monkeypatch.chdir(tmpdir)
    writer = pd.ExcelWriter("designinput_spaces.xlsx", engine="openpyxl")
    MOCK_GENERAL_INPUT.to_excel(
        writer, sheet_name="general_input", index=False, header=None
    )
    mock_spacious_designinput.to_excel(
        writer, sheet_name="designinput", index=False, header=None
    )
    defaultvalues_spacious.to_excel(
        writer, sheet_name="defaultvalues", index=False, header=None
    )
    writer.close()

    dict_design = excel_to_dict("designinput_spaces.xlsx")
    assert next(iter(dict_design["sensitivities"].keys())) == "rms_seed"

    # Check default values parameter names:
    def_params = list(dict_design["defaultvalues"].keys())
    assert [par.strip() for par in def_params] == def_params


def test_mixed_senstype_exception(tmpdir, monkeypatch):
    """Test that exceptions are raised for mixups in user input on types"""
    # pylint: disable=abstract-class-instantiated
    mock_erroneous_designinput = pd.DataFrame(
        data=[
            ["sensname", "numreal", "type", "param_name"],
            ["rms_seed", "", "seed"],
            ["", "", "dist"],
        ]
    )
    monkeypatch.chdir(tmpdir)
    defaultvalues = pd.DataFrame()

    writer = pd.ExcelWriter("designinput4.xlsx", engine="openpyxl")
    MOCK_GENERAL_INPUT.to_excel(
        writer, sheet_name="general_input", index=False, header=None
    )
    mock_erroneous_designinput.to_excel(
        writer, sheet_name="designinput", index=False, header=None
    )
    defaultvalues.to_excel(writer, sheet_name="defaultvalues", index=False, header=None)
    writer.close()

    with pytest.raises(ValueError, match="contains more than one sensitivity type"):
        excel_to_dict("designinput4.xlsx")


def test_has_value():
    """Test a function that is used to check if xlsx-cells are empty or not"""
    assert _has_value(1)
    assert not _has_value(np.nan)

    # This possibly makes no sense, but is the current implementation:
    assert _has_value(None)


def test_background_sheet(tmpdir, monkeypatch):
    """Test loading background values from a sheet"""
    monkeypatch.chdir(tmpdir)
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
        columns=["param_name", "default_value"], data=[["extraseed", "0"]]
    )
    background = pd.DataFrame(
        data=[
            ["param_name", "dist_name", "dist_param1"],
            ["extraseed", "scenario", "30,40,50"],
        ]
    )

    writer = pd.ExcelWriter("designinput.xlsx", engine="openpyxl")
    general_input.to_excel(writer, sheet_name="general_input", index=False, header=None)
    MOCK_DESIGNINPUT.to_excel(
        writer, sheet_name="design_input", index=False, header=None
    )
    defaultvalues.to_excel(writer, sheet_name="defaultvalues", index=False)
    background.to_excel(writer, sheet_name="backgroundsheet", index=False, header=None)
    writer.close()

    dict_design = excel_to_dict("designinput.xlsx")

    # Assert it has been interpreted correctly from input files:
    assert dict_design["background"]["parameters"]["extraseed"] == [
        "scenario",
        ["30,40,50"],
        None,
    ]
    assert dict_design["repeats"] == 3
    assert dict_design["defaultvalues"]["extraseed"] == 0


def test_assert_no_merged_cells(tmpdir, monkeypatch):
    """Test that assert_no_merged_cells detects merged cells"""
    monkeypatch.chdir(tmpdir)

    # Create Excel file with merged cells
    test_data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    writer = pd.ExcelWriter("test_file.xlsx", engine="openpyxl")
    test_data.to_excel(writer, sheet_name="sheet1", index=False)
    writer.close()

    # Add merged cells
    workbook = openpyxl.load_workbook("test_file.xlsx")
    workbook["sheet1"].merge_cells("A1:B1")
    workbook.save("test_file.xlsx")
    workbook.close()

    # Should raise exception
    with pytest.raises(Exception, match="Merged cells"):
        _assert_no_merged_cells("test_file.xlsx")
