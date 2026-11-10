from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest
from ruamel.yaml import YAML

from ert.parameter_design import DesignConfig, FmudesignError, DesignMatrix
from fmudesign import config_to_yaml, excel_to_config
from fmudesign._excel_to_design import (
    _assert_no_merged_cells,
)
from tests.ert.unit_tests.parameter_design._configurations import full_mc_configuration

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
    correlations=None,
    general_sheet="general_input",
    design_sheet="designinput",
    default_sheet="defaultvalues",
    background_sheet="backgroundsheet",
    correlation_sheet="corr1",
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
        if correlations is not None:
            correlations.to_excel(
                writer, sheet_name=correlation_sheet, index=False, header=None
            )
    return path


def test_that_excel_to_design_parses_workbook_into_configuration_dictionary(use_tmpdir):
    input_path = _write_config_workbook("designinput.xlsx")
    design_config = excel_to_config(input_path)

    assert isinstance(design_config, DesignConfig)
    assert design_config.designtype == "onebyone"
    assert design_config.distribution_seed == 42
    assert design_config.defaultvalues == {}
    assert isinstance(design_config.sensitivities, dict)

    sens = design_config.sensitivities
    assert sens["rms_seed"]["seedname"] == "RMS_SEED"
    assert sens["rms_seed"]["senstype"] == "seed"

    alternate_path = _write_config_workbook(
        "designinput2.xlsx",
        general_sheet="Generalinput",
        design_sheet="Design_input",
        default_sheet="DefaultValues",
    )
    design_config = excel_to_config(alternate_path)
    assert isinstance(design_config, DesignConfig)
    assert design_config.sensitivities["rms_seed"]["senstype"] == "seed"

    yaml_file = "dictdesign.yaml"
    config_to_yaml(design_config, yaml_file)
    assert "RMS_SEED" in Path(yaml_file).read_text(encoding="utf-8")


def test_that_duplicate_sensitivity_names_raise_value_error(use_tmpdir):
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
        "designinput.xlsx", design_input=mock_erroneous_designinput
    )

    with pytest.raises(
        FmudesignError, match="Two sensitivities cannot share the same sensname"
    ):
        excel_to_config(input_path)


def test_that_excel_to_design_strips_sensitivity_and_parameter_name_whitespace(
    use_tmpdir,
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
        "designinput.xlsx",
        design_input=mock_spacious_designinput,
        defaultvalues=defaultvalues_spacious,
    )

    design_config = excel_to_config(input_path)
    assert next(iter(design_config.sensitivities.keys())) == "rms_seed"
    assert design_config.defaultvalues == {
        "spacious_multiplier": 1.2,
        "spacious2": 3.3,
    }


def test_that_excel_to_design_rejects_duplicate_default_names_after_trimming(
    use_tmpdir,
):
    input_path = _write_config_workbook(
        "designinput.xlsx",
        defaultvalues=pd.DataFrame(
            [
                ["parametername", "value"],
                ["  a", 1],
                ["a   ", 2],
            ]
        ),
    )

    with pytest.raises(FmudesignError, match="duplicate parameter names"):
        excel_to_config(input_path)


def test_that_mixed_sensitivity_types_raise_value_error(use_tmpdir):
    mock_erroneous_designinput = pd.DataFrame(
        data=[
            ["sensname", "numreal", "type", "param_name"],
            ["rms_seed", "", "seed"],
            ["", "", "dist"],
        ]
    )
    input_path = _write_config_workbook(
        "designinput.xlsx", design_input=mock_erroneous_designinput
    )

    with pytest.raises(FmudesignError, match="contains more than one sensitivity type"):
        excel_to_config(input_path)


@pytest.mark.parametrize(
    ("sens_type", "problem"),
    [
        ("montecarlo", "has invalid type 'montecarlo'"),
        ("", "has no type on its first row"),
    ],
)
def test_that_invalid_sensitivity_type_error_lists_valid_types(
    use_tmpdir, sens_type, problem
):
    design_input = pd.DataFrame(
        data=[["sensname", "numreal", "type", "param_name"], ["Sens_1", "", sens_type]]
    )
    input_path = _write_config_workbook("designinput.xlsx", design_input=design_input)

    with pytest.raises(
        FmudesignError,
        match=(
            rf"Sensitivity 'Sens_1' in the design input sheet {problem}\. "
            r"Valid types are: seed, scenario, dist, ref, background, extern\."
        ),
    ):
        excel_to_config(input_path)


def test_that_excel_to_design_parses_background_sheet(use_tmpdir):
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
        "designinput.xlsx",
        general_input=general_input,
        defaultvalues=defaultvalues,
        background=background,
        design_sheet="design_input",
    )

    design_config = excel_to_config(input_path)

    # Assert it has been interpreted correctly from input files:
    assert design_config.background["parameters"]["extraseed"] == [
        "scenario",
        ["30,40,50"],
        None,
    ]
    assert design_config.repeats == 3
    assert design_config.defaultvalues["extraseed"] == 0


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


@pytest.mark.usefixtures("use_tmpdir")
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
    sheet, missing_field, error
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
    input_path = "designinput.xlsx"

    if sheet == "designinput":
        _write_config_workbook(input_path, design_input=distribution_rows)
    else:
        _write_background_workbook(input_path, distribution_rows, None, sheet)

    with pytest.raises(FmudesignError, match=error) as exc_info:
        excel_to_config(input_path)

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
    use_tmpdir, index, columns, error
):
    corr_matrix = pd.DataFrame(
        [[1.0, np.nan], [0.5, 1.0]],
        index=index,
        columns=columns,
    )
    input_path = _write_background_workbook(
        "designinput.xlsx",
        BACKGROUND_WITH_CORR,
        corr_matrix,
        "backgroundsheet",
    )

    with pytest.raises(FmudesignError, match=error):
        excel_to_config(input_path)


def test_that_missing_correlation_sheet_raises_fmudesign_error(use_tmpdir):
    input_path = _write_background_workbook(
        "designinput.xlsx", BACKGROUND_WITH_CORR, None, "backgroundsheet"
    )

    with pytest.raises(
        FmudesignError,
        match=(
            r"Correlation sheet 'bgcorr' could not be read from 'designinput.xlsx'"
            r".*\nSheets in workbook: \['general_input', 'designinput', "
            r"'defaultvalues', 'backgroundsheet'\]"
        ),
    ):
        excel_to_config(input_path)


def test_that_non_numeric_correlation_entry_raises_fmudesign_error(use_tmpdir):
    corr_matrix = pd.DataFrame(
        [[1.0, np.nan], ["0,5", 1.0]],
        index=["PARAM_A", "PARAM_B"],
        columns=["PARAM_A", "PARAM_B"],
    )
    input_path = _write_background_workbook(
        "designinput.xlsx", BACKGROUND_WITH_CORR, corr_matrix, "backgroundsheet"
    )

    with pytest.raises(
        FmudesignError,
        match=(
            r"All elements in matrix in corr sheet bgcorr must be numbers or "
            r"blank, got: .*'0,5'"
        ),
    ):
        excel_to_config(input_path)


def test_that_missing_background_sheet_error_lists_available_sheets(use_tmpdir):
    input_path = _write_background_workbook(
        "designinput.xlsx", BACKGROUND_WITH_CORR, None, "typo_sheet"
    )

    with pytest.raises(FmudesignError, match="Sheets in workbook") as exc_info:
        excel_to_config(input_path)

    message = str(exc_info.value)
    assert "typo_sheet" in message
    assert "backgroundsheet" in message
    assert "Use 'None' as background" in message


@pytest.mark.parametrize(
    "background_name", ["Backgroundsheet", "background_sheet", " backgroundsheet "]
)
def test_that_background_sheet_matching_ignores_case_underscores_and_whitespace(
    use_tmpdir, background_name
):
    corr_matrix = pd.DataFrame(
        [[1.0, np.nan], [0.5, 1.0]],
        index=["PARAM_A", "PARAM_B"],
        columns=["PARAM_A", "PARAM_B"],
    )
    input_path = _write_background_workbook(
        "designinput.xlsx",
        BACKGROUND_WITH_CORR,
        corr_matrix,
        background_name,
    )

    background = excel_to_config(input_path).background
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
        FmudesignError,
        match=r"Sheet 'missing_background.csv' with background parameters, "
        "specified in the general input sheet, "
        "was not found in 'designinput.xlsx'.",
    ):
        excel_to_config(input_path)


@pytest.mark.parametrize("background_name", ["None", "none", np.nan])
def test_that_none_like_background_names_disable_background(
    use_tmpdir, background_name
):
    input_path = _write_background_workbook(
        "designinput.xlsx", BACKGROUND_WITH_CORR, None, background_name
    )

    assert excel_to_config(input_path).background is None


def test_that_assert_no_merged_cells_rejects_merged_cells(tmp_path):
    input_path = tmp_path / "test_file.xlsx"
    test_data = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    test_data.to_excel(input_path, sheet_name="sheet1", index=False)

    workbook = openpyxl.load_workbook(input_path)
    workbook["sheet1"].merge_cells("A1:B1")
    workbook.save(input_path)
    workbook.close()

    with pytest.raises(FmudesignError, match="Merged cells"):
        _assert_no_merged_cells(input_path)


def test_that_excel_to_design_preserves_seed_strategy(use_tmpdir):
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
        "designinput.xlsx",
        general_input=general,
        design_input=designinput,
    )
    dict_design = excel_to_config(input_path)
    assert dict_design.seed_strategy == "independent"


def test_that_config_to_yaml_does_not_dump_python_specific_content(use_tmpdir):
    design_config = full_mc_configuration(Path("foo.xlsx"))
    config_to_yaml(design_config, "dump.yaml")
    yaml_content = Path("dump.yaml").read_text(encoding="utf-8")
    assert "SeedStrategy" not in yaml_content
    assert "!!python/object" not in yaml_content
    assert "design_config.DesignConfig" not in yaml_content
    assert "_pydantic_" not in yaml_content


def test_that_config_to_roundtrip_recreates_design_config(use_tmpdir):
    design_config = full_mc_configuration(Path("foo.xlsx"))
    config_to_yaml(design_config, "dump.yaml")
    yaml = YAML()
    yaml_content = yaml.load(Path("dump.yaml").read_text(encoding="utf-8"))
    assert design_config == DesignConfig(**yaml_content)


def test_that_empty_rms_seeds_file_raises_value_error(use_tmpdir):
    empty_file = "seeds.txt"
    Path(empty_file).touch()
    general_input = MOCK_GENERAL_INPUT.copy()
    general_input.loc[len(MOCK_GENERAL_INPUT)] = ["rms_seeds", empty_file]

    design_file = "foo.xlsx"
    _write_config_workbook(path=design_file, general_input=general_input)

    with pytest.raises(FmudesignError, match="at least one seed"):
        excel_to_config(design_file)


def test_that_empty_parameter_name_with_decimals_is_dropped_in_config(use_tmpdir):
    wb_path = "design.xlsx"
    design_input = pd.DataFrame(
        [
            ["sensname", "numreal", "type", "param_name", "decimals"],
            ["rms_seed", "", "seed", "", 2],
            ["default", 1, "ref", "foo", 3],
        ]
    )
    _write_config_workbook(wb_path, design_input=design_input)
    config = excel_to_config(wb_path)
    assert config.decimals == {"foo": 3}


@pytest.mark.slow
@pytest.mark.parametrize("correlations", [True, False])
def test_that_generated_distributions_match_configured_statistics(
    use_tmpdir, correlations
):
    NUM_SAMPLES = 10**5

    def gl(paramname, distname, p1, p2, p3="", p4=""):
        """GL = Generate Line. Generates a line in the input sheet."""
        return [
            "",
            pd.NA,
            "",
            paramname,
            "",
            pd.NA,
            "",
            pd.NA,
            distname,
            p1,
            p2,
            p3,
            p4,
            pd.NA,
            "corr1" if correlations else "",
            "",
        ]

    # General input sheet
    general_input = pd.DataFrame(
        data=[
            ["designtype", "onebyone"],
            ["repeats", 1],
            ["rms_seeds", "default"],
            ["background", "None"],
            ["distribution_seed", 42],
        ]
    )

    # Design input sheet
    design_input = pd.DataFrame(
        data=[
            # Normal has params (mean, std, low=-inf, high=inf)
            gl("NORMAL", "normal", 0, 2),
            gl("TRUNCNORM", "normal", 0, 1, -1, 2),
            # Lognormal has params (mean, sigma)
            gl("LOGNORMAL", "logn", 1.5, 0.5),
            gl("TRUNCLOGNORMAL", "logn", 1.5, 0.5, 5, 15),
            # Uniform has params (low, high)
            gl("UNIFORM", "unif", -5, 0),
            # Triangular has params (low, mode, high)
            gl("TRIANG", "triang", -5, 0, 5),
            # Beta has params (alpha, beta, low=0, high=1)
            gl("BETA", "beta", 5, 2),
            gl("SCALEDBETA", "beta", 5, 2, -2, 3),
            # Pert has has params (low, mode, high, scale=4)
            gl("DEFAULTPERT", "pert", -5, 0, 5),
            gl("SCALEPERT", "pert", -5, 0, 5, 1),
            # Loguniform has params (low, high)
            gl("LOGUNIFORM", "logunif", 1, 5),
            # P10/P90 versions
            gl("NORMALP10P90", "normal_p10_p90", -2, 3),
            gl("UNIFORMP10P90", "uniform_p10_p90", -2, 3),
            gl("TRIANGULARP10P90", "triangular_p10_p90", -2, 2, 3),
            gl("PERTP10P90", "pert_p10_p90", -2, 2, 3),
        ],
        columns=[
            "sensname",
            "numreal",
            "type",
            "param_name",
            "senscase1",
            "value1",
            "senscase2",
            "value2",
            "dist_name",
            "dist_param1",
            "dist_param2",
            "dist_param3",
            "dist_param4",
            "decimals",
            "corr_sheet",
            "extern_file",
        ],
    )
    design_input.iloc[0, :3] = ["distr_test", (NUM_SAMPLES), "dist"]

    # Default values sheet
    defaultvalues = pd.DataFrame(
        {
            "param_name": list(design_input["param_name"]),
            "default_value": [0.5] * (len(design_input)),
        }
    )

    # Correlation sheet
    num_vars = len(design_input["param_name"])
    corr_values = np.zeros(shape=(num_vars, num_vars)) + 0.2
    np.fill_diagonal(corr_values, val=1.0)
    upper_idx = np.triu_indices_from(corr_values, k=1)
    # Set upper triangle to blank on the numpy array before creating the DataFrame,
    # since DataFrame.to_numpy() returns a copy in pandas 3 (CoW).
    str_values = corr_values.astype(str)
    str_values[upper_idx] = ""
    corr_sheet = pd.DataFrame(
        str_values,
        columns=list(design_input["param_name"]),
        index=list(design_input["param_name"]),
    )
    # Create a file to do the save => load roundtrip and test that too
    input_path = "designinput.xlsx"
    with pd.ExcelWriter(input_path, engine="openpyxl") as writer:
        general_input.to_excel(
            writer, sheet_name="general_input", index=False, header=None
        )
        design_input.to_excel(writer, sheet_name="designinput", index=False)
        defaultvalues.to_excel(writer, sheet_name="defaultvalues", index=False)
        corr_sheet.to_excel(writer, sheet_name="corr1")

    # Read the file and draw samples
    input_dict = excel_to_config(input_path)
    design = DesignMatrix(input_dict)
    assert len(design.designvalues) == NUM_SAMPLES
    df = design.designvalues

    # Test statistical properties and boundaries of all variables.
    # There were either derived using analytical properties, or empirically
    # by drawing 10 million samples.
    # Tolerance must be high enough to not pick up on rng differences, but low
    # enough to pick up meaningful changes.
    atol = 0.005

    assert np.isclose(df["NORMAL"].mean(), 0.0, atol=atol)
    assert np.isclose(df["NORMAL"].std(), 2.0, atol=atol)

    assert np.isclose(df["TRUNCNORM"].mean(), 0.229637, atol=atol)
    assert np.isclose(df["TRUNCNORM"].std(), 0.720945, atol=atol)
    assert df["TRUNCNORM"].min() >= -1
    assert df["TRUNCNORM"].max() <= 2

    assert np.isclose(df["LOGNORMAL"].mean(), 5.078418, atol=atol)
    assert np.isclose(df["LOGNORMAL"].std(), 2.706487, atol=atol)

    assert df["TRUNCLOGNORMAL"].min() >= 5
    assert df["TRUNCLOGNORMAL"].max() <= 15

    assert df["UNIFORM"].min() >= -5
    assert df["UNIFORM"].max() <= 0
    assert np.isclose(df["UNIFORM"].mean(), -2.5, atol=atol)
    assert np.isclose(df["UNIFORM"].std(), 1.443375, atol=atol)

    assert df["TRIANG"].min() >= -5
    assert df["TRIANG"].max() <= 5
    assert np.isclose(df["TRIANG"].mean(), 0, atol=atol)
    assert np.isclose(df["TRIANG"].std(), 2.041241, atol=atol)

    assert df["BETA"].min() >= 0
    assert df["BETA"].max() <= 1
    assert np.isclose(df["BETA"].mean(), 0.714286, atol=atol)
    assert np.isclose(df["BETA"].std(), 0.159719, atol=atol)

    assert df["SCALEDBETA"].min() >= -2
    assert df["SCALEDBETA"].max() <= 3
    assert np.isclose(df["SCALEDBETA"].mean(), 1.571429, atol=atol)
    assert np.isclose(df["SCALEDBETA"].std(), 0.798596, atol=atol)

    assert df["DEFAULTPERT"].min() >= -5
    assert df["DEFAULTPERT"].max() <= 5
    assert np.isclose(df["DEFAULTPERT"].mean(), 0, atol=atol)
    assert np.isclose(df["DEFAULTPERT"].std(), 1.889822, atol=atol)

    assert df["SCALEPERT"].min() >= -5
    assert df["SCALEPERT"].max() <= 5
    assert np.isclose(df["SCALEPERT"].mean(), 0, atol=atol)
    assert np.isclose(df["SCALEPERT"].std(), 2.5, atol=atol)

    assert np.isclose(df["LOGUNIFORM"].mean(), 2.485339, atol=atol)
    assert np.isclose(df["LOGUNIFORM"].std(), 1.130975, atol=atol)

    # The P10/P90 distributions are all defined to have P10=-2 and P90=3,
    # so we test them by checking that the observed percentiles match
    assert np.isclose(df["NORMALP10P90"].quantile(0.1), -2, atol=atol)
    assert np.isclose(df["NORMALP10P90"].quantile(0.9), 3, atol=atol)

    assert np.isclose(df["UNIFORMP10P90"].quantile(0.1), -2, atol=atol)
    assert np.isclose(df["UNIFORMP10P90"].quantile(0.9), 3, atol=atol)

    assert np.isclose(df["TRIANGULARP10P90"].quantile(0.1), -2, atol=atol)
    assert np.isclose(df["TRIANGULARP10P90"].quantile(0.9), 3, atol=atol)

    assert np.isclose(df["PERTP10P90"].quantile(0.1), -2, atol=atol)
    assert np.isclose(df["PERTP10P90"].quantile(0.9), 3, atol=atol)

    # Check that correlations are close
    if correlations:
        obs_corr = df[design_input["param_name"]].corr().to_numpy()
        assert np.sqrt(np.mean((obs_corr - corr_values) ** 2)) < 0.02
