"""Testing code for generation of design matrices"""

import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ert.shared import __version__ as ert_version
from fmudesign import DesignMatrix, excel_to_dict
from fmudesign import design_distributions as design_dist
from fmudesign._excel_to_dict import _read_defaultvalues
from fmudesign.create_design import MonteCarloSensitivity, _derive_rng
from fmudesign.quality_report import print_corrmat

from ._configurations import (
    background_configuration,
    full_mc_configuration,
    onebyone_configuration,
)


@pytest.mark.slow
@pytest.mark.parametrize("correlations", [True, False])
def test_that_generated_distributions_match_configured_statistics(
    tmp_path, correlations
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
    input_path = tmp_path / "designinput.xlsx"
    with pd.ExcelWriter(input_path, engine="openpyxl") as writer:
        general_input.to_excel(
            writer, sheet_name="general_input", index=False, header=None
        )
        design_input.to_excel(writer, sheet_name="designinput", index=False)
        defaultvalues.to_excel(writer, sheet_name="defaultvalues", index=False)
        corr_sheet.to_excel(writer, sheet_name="corr1")

    # Read the file and draw samples
    input_dict = excel_to_dict(input_path)
    design = DesignMatrix()
    design.generate(input_dict)
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


def test_that_onebyone_design_contains_configured_cases_and_values(tmp_path):
    input_dict = onebyone_configuration()

    # Note that repeats are set to 10 in general_input sheet.
    # So, there are 10 rows for each senscase of type seed and scenario.
    # However, there are 20 rows for multz because numreal is set to 20 in designinput.
    rows_in_design_matrix = 80

    design = DesignMatrix()
    design.generate(input_dict)
    # Checking dimensions of design matrix
    assert design.designvalues.shape == (rows_in_design_matrix, 10)

    output_path = tmp_path / "designmatrix.xlsx"
    design.to_xlsx(str(output_path))
    diskdesign = pd.read_excel(output_path, engine="openpyxl")

    assert (
        diskdesign.columns
        == [
            "REAL",
            "SENSNAME",
            "SENSCASE",
            "RMS_SEED",
            "FAULT_POSITION",
            "DC_MODEL",
            "OWC1",
            "OWC2",
            "OWC3",
            "MULTZ_ILE",
        ]
    ).all()
    assert (diskdesign["REAL"].to_numpy() == np.arange(rows_in_design_matrix)).all()
    ensemble_size = 10
    sensname = (
        ["rms_seed"] * ensemble_size
        + ["faults"] * 2 * ensemble_size  # 2 senscases, east and west
        + ["velmodel"] * ensemble_size
        + ["contacts"] * 2 * ensemble_size  # 2 contacts, shallow and deep
        + ["multz"] * 20
    )
    assert (diskdesign["SENSNAME"] == sensname).all()
    # Sensitivities of type seed like rms_seed automatically get senscase p10_p90,
    # so that P10/P90 is calculated for the tornado plot.
    assert (
        diskdesign[diskdesign["SENSNAME"] == "rms_seed"]["SENSCASE"] == "p10_p90"
    ).all()
    assert (
        diskdesign[diskdesign["SENSNAME"] == "faults"]["SENSCASE"]
        == ["east"] * ensemble_size + ["west"] * ensemble_size
    ).all()
    assert (
        diskdesign[diskdesign["SENSNAME"] == "velmodel"]["SENSCASE"] == "alternative"
    ).all()
    assert (
        diskdesign[diskdesign["SENSNAME"] == "contacts"]["SENSCASE"]
        == ["shallow"] * ensemble_size + ["deep"] * ensemble_size
    ).all()
    assert (
        diskdesign[diskdesign["SENSNAME"] == "multz"]["SENSCASE"] == ["p10_p90"] * 20
    ).all()

    # When rms_seed is set to default it means that RMS_SEED numbers
    # 1000, 1001,... are used.
    # Note that for most senscases, RMS_SEED goes from 1000 to 1009,
    # but that it goes from 1000 to 1019 for multz because numreal
    # is set to 20 in the designinput sheet.
    assert (
        diskdesign["RMS_SEED"]
        == list(range(1000, 1000 + ensemble_size)) * 6 + list(range(1000, 1000 + 20))
    ).all()

    diskdefaults = pd.read_excel(
        output_path, sheet_name="DefaultValues", header=None, engine="openpyxl"
    )
    assert (diskdefaults.columns == [0, 1]).all()
    assert (
        diskdefaults.iloc[:, 0]
        == [
            "RMS_SEED",
            "FAULT_POSITION",
            "DC_MODEL",
            "OWC1",
            "OWC2",
            "OWC3",
            "MULTZ_ILE",
            "PARAM1",
            "PARAM2",
            "PARAM3",
            "PARAM4",
        ]
    ).all()

    diskdefaults = diskdefaults.set_index(0)

    # FAULT_POSITION has two senscases, east with value -1 and west with value 1,
    # so we expect ensemble_size number of rows with -1s and ensemble_size rows with 1s.
    # We expect the remaining rows to be set to the base value
    # set in the defaultvalues sheet.
    fault_position_base = diskdefaults.loc["FAULT_POSITION"].to_list()
    fault_position = (
        fault_position_base * ensemble_size
        + [-1] * ensemble_size
        + [1] * ensemble_size
        + fault_position_base * (rows_in_design_matrix - 3 * ensemble_size)
    )
    assert (diskdesign["FAULT_POSITION"] == fault_position).all()

    dc_model_base = diskdefaults.loc["DC_MODEL"].to_list()
    dc_model = (
        dc_model_base * 3 * ensemble_size
        + ["alternative"] * ensemble_size
        + dc_model_base * (rows_in_design_matrix - 4 * ensemble_size)
    )
    assert (diskdesign["DC_MODEL"] == dc_model).all()

    owc1_base = diskdefaults.loc["OWC1"].to_list()
    owc1 = (
        owc1_base * 4 * ensemble_size
        + [2600] * ensemble_size
        + [2700] * ensemble_size
        + owc1_base * (rows_in_design_matrix - 6 * ensemble_size)
    )
    assert (diskdesign["OWC1"] == owc1).all()

    owc2_base = diskdefaults.loc["OWC2"].to_list()
    owc2 = (
        owc2_base * 4 * ensemble_size
        + [2700] * ensemble_size
        + [2800] * ensemble_size
        + owc2_base * (rows_in_design_matrix - 6 * ensemble_size)
    )
    assert (diskdesign["OWC2"] == owc2).all()

    owc3_base = diskdefaults.loc["OWC3"].to_list()
    owc3 = (
        owc3_base * 4 * ensemble_size
        + [2800] * ensemble_size
        + [2900] * ensemble_size
        + owc3_base * (rows_in_design_matrix - 6 * ensemble_size)
    )
    assert (diskdesign["OWC3"] == owc3).all()

    # MULTZ_ILE contains random numbers so we won't test it here.

    diskmetadata = pd.read_excel(output_path, sheet_name="Metadata", engine="openpyxl")

    assert (diskmetadata.columns == ["Description", "Value"]).all()
    assert diskmetadata["Description"].iloc[0] == "Created using ert version:"
    assert diskmetadata["Value"].iloc[0] == ert_version
    assert diskmetadata["Description"].iloc[1] == "Created on:"

    datetime.fromisoformat(diskmetadata["Value"].iloc[1])


def _assert_design_snapshot(design, snapshot):
    rounded_values = design.designvalues.map(
        lambda value: value if isinstance(value, str) else float(f"{value:.6g}")
    )
    snapshot.assert_match(
        json.dumps(
            {
                "designvalues": rounded_values.to_dict("records"),
                "defaultvalues": dict(design.defaultvalues),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        "design_output_mc_with_correls.json",
    )


def _full_mc_input(tmp_path):
    correlation_path = tmp_path / "full-mc-correlations.xlsx"
    _write_correlation_sheets(
        correlation_path,
        {
            "corr1": (
                ["PARAM1", "PARAM2", "PARAM3"],
                [[1, 0, 0.2], [0, 1, 0], [0.2, 0, 1]],
            ),
            "corr2": (
                ["OWC1", "OWC2", "OWC3"],
                [[1, 0.5, -0.7], [0.5, 1, -0.7], [-0.7, -0.7, 1]],
            ),
            "corr3": (
                ["DATO", "NTG1"],
                [[1, 0.8], [0.8, 1]],
            ),
        },
    )
    return full_mc_configuration(correlation_path)


def test_that_joint_full_monte_carlo_design_matches_snapshot(snapshot, tmp_path):
    input_dict = _full_mc_input(tmp_path)
    design = DesignMatrix()
    design.generate(input_dict)

    _assert_design_snapshot(design, snapshot)


def test_that_independent_full_monte_carlo_design_matches_snapshot(snapshot, tmp_path):
    input_dict = _full_mc_input(tmp_path)
    input_dict["seed_strategy"] = "independent"
    design = DesignMatrix()
    design.generate(input_dict)

    _assert_design_snapshot(design, snapshot)


def test_that_full_monte_carlo_design_applies_dependencies_and_correlations(tmp_path):
    input_dict = _full_mc_input(tmp_path)

    design = DesignMatrix()
    design.generate(input_dict)

    design_values = design.designvalues
    assert design_values.shape == (500, 16)

    # Make sure adding dependent discrete parameters works.
    expected_derived_1 = {
        "2018-11-02": 1,
        "2018-11-03": 2,
        "2018-11-04": 3,
    }
    expected_derived_2 = {
        "2018-11-02": "a",
        "2018-11-03": "b",
        "2018-11-04": "c",
    }
    assert design_values["DERIVED_PARAM1"].tolist() == (
        design_values["DATO"].map(expected_derived_1).tolist()
    )
    assert design_values["DERIVED_PARAM2"].tolist() == (
        design_values["DATO"].map(expected_derived_2).tolist()
    )

    # Check that variables are correlated using Pearson correlation
    # Using 95% confidence intervals for correlation coefficients.
    #
    # The confidence interval calculation assumes:
    #   - Large sample size (n > 30)
    #   - Bivariate normal distribution of variables
    #   - Linear relationship between variables
    # When these assumptions are violated (e.g. with skewed distributions),
    # the intervals become less reliable
    r_obj = stats.pearsonr(design_values["OWC1"], design_values["OWC2"])
    r_ci = r_obj.confidence_interval(confidence_level=0.95)
    assert r_ci[0] <= 0.5 <= r_ci[1]

    r_obj = stats.pearsonr(design_values["OWC2"], design_values["OWC3"])
    r_ci = r_obj.confidence_interval(confidence_level=0.95)
    assert r_ci[0] <= -0.7 <= r_ci[1]

    r_obj = stats.pearsonr(design_values["PARAM1"], design_values["PARAM2"])
    r_ci = r_obj.confidence_interval(confidence_level=0.95)
    assert r_ci[0] <= 0 <= r_ci[1]

    # Using wide tolerance because the non-linear transformation between normal
    # and target distributions can alter correlation strength.
    assert np.isclose(
        stats.spearmanr(design_values["PARAM1"], design_values["PARAM3"])[0],
        0.2,
        atol=0.1,
    )

    # Check that we can add correlations to discrete variables.
    # DATO is stored as strings, so convert to ordinals: spearmanr needs
    # numeric input, otherwise scipy passes an object array to np.cov.
    dato_ordinal = pd.to_datetime(design_values["DATO"]).astype("int64")
    assert np.isclose(
        stats.spearmanr(dato_ordinal, design_values["NTG1"])[0], 0.8, atol=0.1
    )

    date_fractions = design_values["DATO"].value_counts(normalize=True)
    assert math.isclose(date_fractions.loc["2018-11-02"], 0.3)
    assert math.isclose(date_fractions.loc["2018-11-03"], 0.4)
    assert math.isclose(date_fractions.loc["2018-11-04"], 0.3)


@pytest.mark.slow
def test_that_background_fills_inactive_parameters_without_overwriting_sensitivities(
    tmp_path,
):
    correlation_path = tmp_path / "background-correlations.xlsx"
    _write_correlation_sheets(
        correlation_path,
        {
            "background_corr": (
                ["PARAM17", "PARAM18", "PARAM19"],
                [[1, 0.9, 0], [0.9, 1, 0.9], [0, 0.9, 1]],
            ),
            "corr0": (
                ["PARAM5", "PARAM6"],
                [[1, 0.8], [0.8, 1]],
            ),
            "corr1": (
                ["PARAM9", "PARAM10", "PARAM11", "PARAM12"],
                [
                    [1, 0.9, 0, 0],
                    [0.9, 1, 0.9, 0],
                    [0, 0.9, 1, 0],
                    [0, 0, 0, 1],
                ],
            ),
        },
    )
    external_parameters = tmp_path / "external-parameters.csv"
    pd.DataFrame(
        {
            "PARAM13": range(11),
            "PARAM14": range(1, 12),
            "PARAM15": range(2, 13),
            "PARAM16": pd.date_range("2018-11-01", periods=11).strftime("%Y-%m-%d"),
        }
    ).to_csv(external_parameters, index=False)
    input_dict = background_configuration(correlation_path, external_parameters)
    input_dict["distribution_seed"] = 42

    design = DesignMatrix()
    design.generate(input_dict)

    background_params = ["PARAM17", "PARAM18", "PARAM19"]
    background_vals = design.designvalues.loc[
        design.designvalues["SENSNAME"] == "background", background_params
    ]
    velmodel_vals = design.designvalues.loc[
        design.designvalues["SENSNAME"] == "velmodel", background_params
    ]
    assert (background_vals.to_numpy() == velmodel_vals.to_numpy()).all()

    # Background samples fill inactive parameters, but must not replace values
    # explicitly sampled by a sensitivity using the same parameter names.
    sens9_vals = design.designvalues.loc[
        design.designvalues["SENSNAME"] == "sens9", background_params
    ]
    assert design.backgroundvalues is not None
    sampled_background = design.backgroundvalues[background_params]
    assert sens9_vals.shape == sampled_background.shape
    for parameter in background_params:
        assert not np.array_equal(
            sens9_vals[parameter].to_numpy(),
            sampled_background[parameter].to_numpy(),
        ), f"sens9 values for {parameter} were replaced by background samples"

    faults_vals = design.designvalues.loc[
        design.designvalues["SENSNAME"] == "faults", background_params
    ]
    contacts_vals = design.designvalues.loc[
        design.designvalues["SENSNAME"] == "contacts", background_params
    ]
    assert (faults_vals.to_numpy() == contacts_vals.to_numpy()).all()

    sens6 = design.designvalues[design.designvalues["SENSNAME"] == "sens6"]
    assert np.isclose(
        stats.spearmanr(sens6["PARAM5"], sens6["PARAM6"])[0],
        0.8,
        atol=0.1,
    )
    sens7 = design.designvalues[design.designvalues["SENSNAME"] == "sens7"]
    assert np.isclose(
        stats.spearmanr(sens7["PARAM9"], sens7["PARAM10"])[0],
        0.8,
        atol=0.2,
    )
    assert np.isclose(
        stats.spearmanr(sens7["PARAM10"], sens7["PARAM11"])[0],
        0.8,
        atol=0.2,
    )


def test_that_read_defaultvalues_rejects_names_duplicated_after_stripping_whitespace(
    tmp_path,
):
    defaultvalues = pd.DataFrame(
        columns=["param_name", "default_value"],
        data=[
            ["a", 1.0],
            ["b", 2.0],
            [" a", 3.0],  # Should be treated as duplicate of "a" after stripping
            ["c", 4.0],
            ["c  ", 5.0],  # Should be treated as duplicate of "c" after stripping
        ],
    )

    input_path = tmp_path / "test_defaults.xlsx"
    defaultvalues.to_excel(input_path, sheet_name="defaultvalues", index=False)

    with pytest.raises(
        ValueError,
        match=(
            "Duplicate parameter names found in sheet "
            r"'defaultvalues': a, c\. All parameter names must be unique\."
        ),
    ):
        _read_defaultvalues(input_path, "defaultvalues")


def _write_correlation_excel(filepath, names, lower_values):
    """Helper: write a correlation sheet with given lower-triangular values."""
    n = len(names)
    arr = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1):
            arr[i, j] = lower_values[i][j]
    _write_correlation_sheets(filepath, {"corr1": (names, arr)})


def test_that_read_correlations_returns_labeled_symmetric_matrix(tmp_path):
    names = ["A", "B", "C"]
    lower = [[1.0], [0.5, 1.0], [0.3, 0.4, 1.0]]
    filepath = tmp_path / "corr.xlsx"
    _write_correlation_excel(filepath, names, lower)

    result = design_dist.read_correlations(str(filepath), corr_sheet="corr1")
    arr = result.to_numpy()

    assert list(result.index) == names
    assert list(result.columns) == names
    np.testing.assert_array_almost_equal(arr, arr.T)
    np.testing.assert_array_almost_equal(np.diag(arr), [1.0, 1.0, 1.0])
    assert np.isclose(arr[1, 0], 0.5)
    assert np.isclose(arr[0, 1], 0.5)


def test_that_print_corrmat_formats_without_mutating_input(capsys):
    values = np.array([[1, -0.0, 0.9], [-0.0, 1, 0], [0.9, 0, 1.0]])
    df = pd.DataFrame(values, index=["A", "B", "C"], columns=["A", "B", "C"])
    original = df.copy()

    print_corrmat(df)

    output = capsys.readouterr().out
    assert "1.00" in output
    assert ".90" in output
    assert "-0.00" not in output
    pd.testing.assert_frame_equal(df, original)


def test_that_fill_with_background_values_replaces_missing_parameter_values():
    dm = DesignMatrix()
    dm.designvalues = pd.DataFrame(
        {
            "SENSNAME": ["s1", "s1"],
            "SENSCASE": ["c1", "c1"],
            "param1": [np.nan, np.nan],
        }
    )
    dm.backgroundvalues = pd.DataFrame({"param1": [10.0, 20.0]})
    dm._fill_with_background_values()

    assert "index" not in dm.designvalues.columns
    assert dm.designvalues["param1"].tolist() == [10.0, 20.0]


def test_that_set_decimals_propagates_zero_precision_to_dependency():
    design = DesignMatrix()
    design.designvalues = pd.DataFrame({"SOURCE": [1.6], "TARGET": [1.6]})
    config = {
        "decimals": {"SOURCE": 0},
        "sensitivities": {
            "sens": {
                "dependencies": {
                    "SOURCE": {"to_params": {"TARGET": []}},
                }
            }
        },
    }

    design._set_decimals(config)

    assert design.designvalues["TARGET"].tolist() == [2.0]


def _sample_mc(
    params,
    *,
    corrdict=None,
    strategy="independent",
    base=42,
    size=200,
    correlation_iterations=0,
    sensname="foo",
):
    """Sample a MonteCarloSensitivity and return the resulting dataframe."""
    sens = MonteCarloSensitivity(sensname=sensname, verbosity=0)
    sens.generate(
        size=size,
        parameters=params,
        seedvalues=None,
        corrdict=corrdict,
        rng=np.random.default_rng(0),
        correlation_iterations=correlation_iterations,
        seed_strategy=strategy,
        base_seed=base,
    )
    return sens.sensvalues


def _col(df, name):
    return df[name].to_numpy(dtype=float)


def _write_correlation_sheets(path, sheets):
    """Write correlation matrices to an xlsx, blanking the upper triangle as
    required by read_correlations. sheets maps sheet_name -> (names, matrix).
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, (names, matrix) in sheets.items():
            frame = pd.DataFrame(
                np.array(matrix, dtype=float), index=names, columns=names
            )
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    frame.iloc[i, j] = np.nan
            frame.to_excel(writer, sheet_name=sheet)


def _design_dict(
    params, strategy, *, repeats=10, distribution_seed=42, background=None
):
    defaultvalues = dict.fromkeys(params, 0)
    if background is not None:
        defaultvalues.update(dict.fromkeys(background["parameters"], 0))
    config = {
        "designtype": "onebyone",
        "seeds": None,
        "repeats": repeats,
        "distribution_seed": distribution_seed,
        "seed_strategy": strategy,
        "defaultvalues": defaultvalues,
        "sensitivities": {
            "s1": {"senstype": "dist", "parameters": params, "correlations": None}
        },
    }
    if background is not None:
        config["background"] = background
    return config


def test_that_omitted_seed_strategy_uses_joint_sampling():
    params = {"A": ("normal", ["0", "1"], None), "B": ("uniform", ["0", "1"], None)}

    def sample(**kwargs):
        sens = MonteCarloSensitivity(sensname="foo", verbosity=0)
        sens.generate(
            size=8,
            parameters=params,
            seedvalues=None,
            corrdict=None,
            rng=np.random.default_rng(7),
            correlation_iterations=0,
            **kwargs,
        )
        return _col(sens.sensvalues, "A")

    np.testing.assert_array_equal(
        sample(), sample(seed_strategy="joint", base_seed=None)
    )


def _assert_stability(stable, before, after, columns):
    """Assert the given columns are bit-identical (stable) or changed."""
    for col in columns:
        first, second = _col(before, col), _col(after, col)
        if stable:
            np.testing.assert_array_equal(first, second, err_msg=col)
        else:
            assert not np.array_equal(first, second), col


@pytest.mark.parametrize("strategy", ["joint", "independent"])
def test_that_sampling_strategies_reproduce_configured_marginal_distributions(
    strategy,
):
    params = {"N": ("normal", ["0", "2"], None), "U": ("uniform", ["-5", "0"], None)}
    design = DesignMatrix()
    design.generate(_design_dict(params, strategy, repeats=10000))

    n = design.designvalues["N"].to_numpy(float)
    u = design.designvalues["U"].to_numpy(float)
    assert np.isclose(n.mean(), 0.0, atol=0.1)
    assert np.isclose(n.std(), 2.0, atol=0.1)
    assert u.min() >= -5
    assert u.max() <= 0
    assert np.isclose(u.mean(), -2.5, atol=0.1)


def test_that_independent_sampling_preserves_values_when_parameters_are_reordered():
    ab = {"A": ("normal", ["0", "1"], None), "B": ("uniform", ["0", "1"], None)}
    ba = {"B": ("uniform", ["0", "1"], None), "A": ("normal", ["0", "1"], None)}
    _assert_stability(
        stable=True, before=_sample_mc(ab), after=_sample_mc(ba), columns=["A", "B"]
    )


def test_that_independent_sampling_preserves_existing_values_when_parameter_is_added():
    base = {
        "A": ("normal", ["0", "1"], None),
        "D": ("discrete", ["red, green, blue"], None),
    }
    added = {**base, "C": ("uniform", ["0", "1"], None)}
    first, second = _sample_mc(base), _sample_mc(added)
    _assert_stability(stable=True, before=first, after=second, columns=["A"])
    assert list(first["D"]) == list(second["D"])


def test_that_independent_sampling_preserves_other_values_when_distribution_changes():
    base = {"A": ("normal", ["0", "1"], None), "B": ("uniform", ["0", "1"], None)}
    changed = {"A": ("normal", ["0", "1"], None), "B": ("uniform", ["0", "5"], None)}
    np.testing.assert_array_equal(
        _col(_sample_mc(base), "A"),
        _col(_sample_mc(changed), "A"),
    )


def test_that_independent_sampling_changes_values_when_base_seed_changes():
    base = {"A": ("normal", ["0", "1"], None)}
    assert not np.allclose(
        _col(_sample_mc(base, base=1), "A"), _col(_sample_mc(base, base=2), "A")
    )


@pytest.mark.parametrize("strategy", ["joint", "independent"])
def test_that_sampling_strategies_induce_configured_group_correlation(
    tmp_path, strategy
):
    path = tmp_path / "corr.xlsx"
    _write_correlation_sheets(path, {"corr1": (["X", "Y"], [[1.0, 0.7], [0.7, 1.0]])})
    params = {
        "X": ("normal", ["0", "1"], "corr1"),
        "Y": ("normal", ["0", "1"], "corr1"),
    }
    corrdict = {"inputfile": str(path), "sheetnames": ["corr1"]}
    result = _sample_mc(
        params,
        corrdict=corrdict,
        strategy=strategy,
        size=5000,
        correlation_iterations=0,
    )
    achieved = np.corrcoef(_col(result, "X"), _col(result, "Y"))[0, 1]
    assert abs(achieved - 0.7) < 0.05


def test_that_independent_sampling_preserves_other_groups_when_group_is_removed(
    tmp_path,
):
    path = tmp_path / "corr.xlsx"
    _write_correlation_sheets(
        path,
        {
            "corr1": (["X", "Y"], [[1.0, 0.7], [0.7, 1.0]]),
            "corr2": (["P", "Q"], [[1.0, 0.5], [0.5, 1.0]]),
        },
    )
    corrdict = {"inputfile": str(path), "sheetnames": ["corr1", "corr2"]}
    full = {
        "X": ("normal", ["0", "1"], "corr1"),
        "Y": ("normal", ["0", "1"], "corr1"),
        "P": ("normal", ["0", "1"], "corr2"),
        "Q": ("normal", ["0", "1"], "corr2"),
        "U": ("uniform", ["0", "1"], None),
    }
    without_group1 = {
        "P": ("normal", ["0", "1"], "corr2"),
        "Q": ("normal", ["0", "1"], "corr2"),
        "U": ("uniform", ["0", "1"], None),
    }
    r_full = _sample_mc(full, corrdict=corrdict)
    r_wo = _sample_mc(without_group1, corrdict=corrdict)
    for name in ("P", "Q", "U"):
        np.testing.assert_array_equal(
            _col(r_full, name), _col(r_wo, name), err_msg=name
        )


def test_that_independent_sampling_preserves_background_when_parameter_is_added():
    """Background parameters are seeded independently too, so adding one leaves
    the others unchanged (the joint strategy would reshuffle them).
    """
    bg2 = {
        "parameters": {
            "BG1": ("normal", ["0", "1"], None),
            "BG2": ("uniform", ["0", "1"], None),
        },
        "correlations": None,
    }
    bg3 = {
        "parameters": {**bg2["parameters"], "BG3": ("triang", ["0", "1", "2"], None)},
        "correlations": None,
    }
    params = {"A": ("normal", ["0", "1"], None)}
    d2, d3 = DesignMatrix(), DesignMatrix()
    d2.generate(_design_dict(params, "independent", background=bg2))
    d3.generate(_design_dict(params, "independent", background=bg3))
    np.testing.assert_array_equal(
        d2.designvalues["BG1"].to_numpy(float), d3.designvalues["BG1"].to_numpy(float)
    )
    np.testing.assert_array_equal(
        d2.designvalues["BG2"].to_numpy(float), d3.designvalues["BG2"].to_numpy(float)
    )


def test_that_independent_sampling_without_seed_is_finite_and_nonreproducible():
    """With no distribution_seed a random base is drawn per run: the output must
    be valid but differ between runs.
    """
    params = {"A": ("normal", ["0", "1"], None), "B": ("uniform", ["0", "1"], None)}
    d1, d2 = DesignMatrix(), DesignMatrix()
    d1.generate(_design_dict(params, "independent", distribution_seed=None))
    d2.generate(_design_dict(params, "independent", distribution_seed=None))
    a1 = d1.designvalues["A"].to_numpy(float)
    a2 = d2.designvalues["A"].to_numpy(float)
    assert not np.isnan(a1).any()
    assert not np.allclose(a1, a2)


@pytest.mark.parametrize(
    ("strategy", "stable"), [("independent", True), ("joint", False)]
)
def test_that_adding_parameter_preserves_values_only_with_independent_strategy(
    strategy, stable
):
    p2 = {"A": ("normal", ["0", "1"], None), "B": ("uniform", ["0", "1"], None)}
    p3 = {**p2, "C": ("triang", ["0", "1", "2"], None)}
    d2, d3 = DesignMatrix(), DesignMatrix()
    d2.generate(_design_dict(p2, strategy))
    d3.generate(_design_dict(p3, strategy))
    _assert_stability(stable, d2.designvalues, d3.designvalues, ["A", "B"])


def test_that_derive_rng_distinguishes_different_key_component_boundaries():
    keys_a = ("a", "param", "b:param:c")
    keys_b = ("a:param:b", "param", "c")
    stream_a = _derive_rng(42, *keys_a).random(8)
    stream_b = _derive_rng(42, *keys_b).random(8)
    assert not np.array_equal(stream_a, stream_b)


def test_that_derive_rng_repeats_stream_for_same_key():
    np.testing.assert_array_equal(
        _derive_rng(42, "foo", "param", "A").random(8),
        _derive_rng(42, "foo", "param", "A").random(8),
    )


def test_that_independent_sampling_allows_shared_group_and_parameter_name(tmp_path):
    path = tmp_path / "corr.xlsx"
    _write_correlation_sheets(path, {"PORO": (["X", "Y"], [[1.0, 0.7], [0.7, 1.0]])})
    params = {
        "X": ("normal", ["0", "1"], "PORO"),
        "Y": ("normal", ["0", "1"], "PORO"),
        "PORO": ("normal", ["0", "1"], None),
    }
    result = _sample_mc(
        params,
        corrdict={"inputfile": str(path), "sheetnames": ["PORO"]},
        size=5000,
    )
    x, y = _col(result, "X"), _col(result, "Y")
    poro = _col(result, "PORO")
    assert abs(np.corrcoef(x, y)[0, 1] - 0.7) < 0.05
    assert abs(np.corrcoef(x, poro)[0, 1]) < 0.05


@pytest.mark.parametrize("strategy", ["joint", "independent"])
def test_that_overlapping_correlation_groups_raise_value_error(tmp_path, strategy):
    """A parameter listed in two correlation matrices is ambiguous: only one of
    the two requested correlations can be honoured, so it must be rejected.

    'Z' is the offending parameter: it is assigned to 'corr2', but is also
    listed in the matrix of 'corr1'. All four parameters are needed, since a
    sheet with only one parameter assigned to it is skipped as uncorrelated.
    """
    path = tmp_path / "corr.xlsx"
    _write_correlation_sheets(
        path,
        {
            "corr1": (["X", "Y", "Z"], [[1, 0.8, 0.8], [0.8, 1, 0.8], [0.8, 0.8, 1]]),
            "corr2": (["Z", "W"], [[1, 0.8], [0.8, 1]]),
        },
    )
    params = {
        "X": ("normal", ["0", "1"], "corr1"),
        "Y": ("normal", ["0", "1"], "corr1"),
        "Z": ("normal", ["0", "1"], "corr2"),
        "W": ("normal", ["0", "1"], "corr2"),
    }
    with pytest.raises(
        ValueError,
        match=(
            "Parameter 'Z' is part of several correlation groups: 'corr1' and 'corr2'"
        ),
    ):
        _sample_mc(
            params,
            corrdict={"inputfile": str(path), "sheetnames": ["corr1", "corr2"]},
            strategy=strategy,
            size=50,
        )


def test_that_unknown_seed_strategy_raises_value_error():
    """The low-level API must reject typos instead of silently using 'joint'."""
    with pytest.raises(ValueError, match="seed_strategy"):
        _sample_mc({"A": ("normal", ["0", "1"], None)}, strategy="bogus")


def test_that_independent_sampling_without_base_seed_raises_value_error():
    with pytest.raises(ValueError, match="base_seed"):
        _sample_mc({"A": ("normal", ["0", "1"], None)}, base=None)


def test_that_independent_sampling_uses_distinct_stream_per_sensitivity():
    """The sensitivity name is part of the key, so two sensitivities sharing a
    parameter name must not receive identical samples.
    """
    params = {"A": ("normal", ["0", "1"], None)}
    first = _sample_mc(params, sensname="sens1")
    second = _sample_mc(params, sensname="sens2")
    assert not np.allclose(_col(first, "A"), _col(second, "A"))


def test_that_independent_sampling_preserves_correlated_background_when_extended(
    tmp_path,
):
    """Background parameters may also be correlated; the group must keep its
    correlation and stay stable when an unrelated background parameter is added.
    """
    path = tmp_path / "corr.xlsx"
    _write_correlation_sheets(
        path, {"bgcorr": (["BG1", "BG2"], [[1.0, 0.6], [0.6, 1.0]])}
    )
    corr_params = {
        "BG1": ("normal", ["0", "1"], "bgcorr"),
        "BG2": ("normal", ["0", "1"], "bgcorr"),
    }
    corrdict = {"inputfile": str(path), "sheetnames": ["bgcorr"]}
    background = {"parameters": corr_params, "correlations": corrdict}
    extended = {
        "parameters": {**corr_params, "BG3": ("uniform", ["0", "1"], None)},
        "correlations": corrdict,
    }
    params = {"A": ("normal", ["0", "1"], None)}

    d1, d2 = DesignMatrix(), DesignMatrix()
    d1.generate(
        _design_dict(params, "independent", repeats=2000, background=background)
    )
    d2.generate(_design_dict(params, "independent", repeats=2000, background=extended))

    bg1 = d1.designvalues["BG1"].to_numpy(float)
    bg2 = d1.designvalues["BG2"].to_numpy(float)
    assert abs(np.corrcoef(bg1, bg2)[0, 1] - 0.6) < 0.05
    np.testing.assert_array_equal(bg1, d2.designvalues["BG1"].to_numpy(float))
    np.testing.assert_array_equal(bg2, d2.designvalues["BG2"].to_numpy(float))
