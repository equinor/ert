"""Testing code for generation of design matrices"""

import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import semeio
from semeio.fmudesign import DesignMatrix, excel_to_dict
from semeio.fmudesign import design_distributions as design_dist
from semeio.fmudesign._excel_to_dict import _read_defaultvalues
from semeio.fmudesign.quality_report import print_corrmat

TESTDATA = Path(__file__).parent / "data"


@pytest.mark.parametrize("correlations", [True, False])
def test_distribution_statistis(tmpdir, monkeypatch, correlations):
    """This test ensures that if any large-sample statistics for any distribution
    changes, we will likely pick it up in the future."""

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
    FILENAME = "designinput.xlsx"
    with pd.ExcelWriter(FILENAME, engine="openpyxl") as writer:
        general_input.to_excel(
            writer, sheet_name="general_input", index=False, header=None
        )
        design_input.to_excel(writer, sheet_name="designinput", index=False)
        defaultvalues.to_excel(writer, sheet_name="defaultvalues", index=False)
        corr_sheet.to_excel(writer, sheet_name="corr1")

    # Read the file and draw samples
    input_dict = excel_to_dict(FILENAME)
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


def test_generate_onebyone(tmpdir):
    """Test generation of onebyone design"""

    inputfile = TESTDATA / "config/design_input_example1.xlsx"

    input_dict = excel_to_dict(inputfile)

    # Note that repeats are set to 10 in general_input sheet.
    # So, there are 10 rows for each senscase of type seed and scenario.
    # However, there are 20 rows for multz because numreal is set to 20 in designinput.
    rows_in_design_matrix = 80

    design = DesignMatrix()
    design.generate(input_dict)
    # Checking dimensions of design matrix
    assert design.designvalues.shape == (rows_in_design_matrix, 10)

    # Write to disk and check some validity
    tmpdir.chdir()
    design.to_xlsx("designmatrix.xlsx")
    assert Path("designmatrix.xlsx").exists
    diskdesign = pd.read_excel("designmatrix.xlsx", engine="openpyxl")

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
        "designmatrix.xlsx", sheet_name="DefaultValues", header=None, engine="openpyxl"
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

    diskmetadata = pd.read_excel(
        "designmatrix.xlsx", sheet_name="Metadata", engine="openpyxl"
    )

    assert (diskmetadata.columns == ["Description", "Value"]).all()
    assert diskmetadata["Description"].iloc[0] == "Created using semeio version:"
    assert diskmetadata["Value"].iloc[0] == semeio.__version__
    assert diskmetadata["Description"].iloc[1] == "Created on:"

    # For the timestamp, we can't check the exact value since it will differ
    # each time the test runs. Instead, we can verify it's a valid datetime string
    # by attempting to parse it
    timestamp = diskmetadata["Value"].iloc[1]
    try:
        datetime.fromisoformat(timestamp)
    except ValueError:
        pytest.fail("Timestamp in Metadata sheet is not in expected format")


def test_generate_full_mc_snapshot(snapshot):
    """Test that full monte carlo design matrix generation remains consistent.

    This is a snapshot test that verifies the entire output of the design matrix
    generation process, including both the design values and default values.
    """
    # Setup
    inputfile = TESTDATA / "config/design_input_mc_with_correls.xlsx"
    input_dict = excel_to_dict(inputfile)
    design = DesignMatrix()

    # Generate the design matrix
    design.generate(input_dict)

    # Round to N significant figures
    df_rounded = design.designvalues.map(
        lambda x: x if isinstance(x, str) else float(f"{x:.6g}")
    )

    # Prepare data for snapshot comparison
    snapshot_dict = {
        "designvalues": df_rounded.to_dict("records"),
        "defaultvalues": dict(design.defaultvalues),
    }

    # Serialize to string for snapshot comparison
    snapshot_str = json.dumps(
        snapshot_dict,
        indent=2,
        sort_keys=True,
    )

    # Verify against snapshot
    snapshot.assert_match(snapshot_str, "design_output_mc_with_correls.json")


def test_generate_full_mc(tmpdir):
    """Test generation of full monte carlo"""
    inputfile = TESTDATA / "config/design_input_mc_with_correls.xlsx"
    input_dict = excel_to_dict(inputfile)

    design = DesignMatrix()
    design.generate(input_dict)

    # Checking dimensions of design matrix
    assert design.designvalues.shape == (500, 16)

    # Write to disk and check some validity
    tmpdir.chdir()
    design.to_xlsx("designmatrix.xlsx")
    assert Path("designmatrix.xlsx").exists
    diskdesign = pd.read_excel(
        "designmatrix.xlsx", sheet_name="DesignSheet01", engine="openpyxl"
    )
    assert "REAL" in diskdesign
    assert "SENSNAME" in diskdesign
    assert "SENSCASE" in diskdesign
    assert not diskdesign.empty

    diskdefaults = pd.read_excel(
        "designmatrix.xlsx", sheet_name="DefaultValues", engine="openpyxl"
    )
    assert not diskdefaults.empty
    assert len(diskdefaults.columns) == 2

    # Make sure adding dependent discrete parameters works.
    disk_depends = pd.read_excel(inputfile, sheet_name="depend1", engine="openpyxl")
    df_merged = diskdesign.merge(disk_depends, on="DATO", how="inner")
    assert (df_merged["DERIVED_PARAM1_x"] == df_merged["DERIVED_PARAM1_y"]).all()
    assert (df_merged["DERIVED_PARAM2_x"] == df_merged["DERIVED_PARAM2_y"]).all()

    # Check that variables are correlated using Pearson correlation
    # Using 95% confidence intervals for correlation coefficients.
    #
    # The confidence interval calculation assumes:
    #   - Large sample size (n > 30)
    #   - Bivariate normal distribution of variables
    #   - Linear relationship between variables
    # When these assumptions are violated (e.g. with skewed distributions),
    # the intervals become less reliable
    r_obj = stats.pearsonr(diskdesign["OWC1"], diskdesign["OWC2"])
    r_ci = r_obj.confidence_interval(confidence_level=0.95)
    assert r_ci[0] <= 0.5 <= r_ci[1]

    r_obj = stats.pearsonr(diskdesign["OWC2"], diskdesign["OWC3"])
    r_ci = r_obj.confidence_interval(confidence_level=0.95)
    assert r_ci[0] <= -0.7 <= r_ci[1]

    r_obj = stats.pearsonr(diskdesign["PARAM1"], diskdesign["PARAM2"])
    r_ci = r_obj.confidence_interval(confidence_level=0.95)
    assert r_ci[0] <= 0 <= r_ci[1]

    # Using wide tolerance because the non-linear transformation between normal
    # and target distributions can alter correlation strength.
    assert np.isclose(
        stats.spearmanr(diskdesign["PARAM1"], diskdesign["PARAM3"])[0], 0.2, atol=0.1
    )

    # Check that we can add correlations to discrete variables.
    # DATO is stored as strings, so convert to ordinals: spearmanr needs
    # numeric input, otherwise scipy passes an object array to np.cov.
    dato_ordinal = pd.to_datetime(diskdesign["DATO"]).astype("int64")
    assert np.isclose(
        stats.spearmanr(dato_ordinal, diskdesign["NTG1"])[0], 0.8, atol=0.1
    )

    date_fractions = diskdesign["DATO"].value_counts(normalize=True)
    assert math.isclose(date_fractions.loc["2018-11-02"], 0.3)
    assert math.isclose(date_fractions.loc["2018-11-03"], 0.4)
    assert math.isclose(date_fractions.loc["2018-11-04"], 0.3)


def test_generate_background(tmpdir):
    inputfile = TESTDATA / "config/design_input_background.xlsx"
    input_dict = excel_to_dict(inputfile)
    source_file = TESTDATA / "config/doe1.xlsx"
    dest_file = tmpdir.join("doe1.xlsx")
    shutil.copy2(source_file, dest_file)

    with tmpdir.as_cwd():
        design = DesignMatrix()
        design.generate(input_dict)

        # Check that background parameters have same values in different sensitivities.
        background_params = ["PARAM17", "PARAM18", "PARAM19"]

        background_vals = design.designvalues.loc[
            design.designvalues["SENSNAME"] == "background", background_params
        ]
        velmodel_vals = design.designvalues.loc[
            design.designvalues["SENSNAME"] == "velmodel", background_params
        ]

        assert (background_vals.to_numpy() == velmodel_vals.to_numpy()).all()

        faults_vals = design.designvalues.loc[
            design.designvalues["SENSNAME"] == "faults", background_params
        ]
        contacts_vals = design.designvalues.loc[
            design.designvalues["SENSNAME"] == "contacts", background_params
        ]

        assert (faults_vals.to_numpy() == contacts_vals.to_numpy()).all()

        sens6 = design.designvalues[design.designvalues["SENSNAME"] == "sens6"]
        # PARAM5 ~ TruncatedNormal(3, 1, 1, 5)
        # PARAM6 ~ Uniform(0, 1)
        assert np.isclose(
            stats.spearmanr(sens6["PARAM5"], sens6["PARAM6"])[0],
            0.8,
            atol=0.1,
        )
        sens7 = design.designvalues[design.designvalues["SENSNAME"] == "sens7"]

        # PARAM9 and PARAM10 have a target correlation of 0.9 in the design config.
        # The input correlation matrix is not positive semi-definite and is
        # transformed to the closest positive semi-definite correlation matrix.
        # The new correlation coefficient is 0.8.
        # Using wide tolerance because the non-linear transformation between normal
        # and target distributions can alter correlation strength.
        assert np.isclose(
            stats.spearmanr(sens7["PARAM9"], sens7["PARAM10"])[0],
            0.8,
            atol=0.2,
        )

        assert np.isclose(
            stats.spearmanr(sens7["PARAM10"], sens7["PARAM11"])[0],
            0.8,
            atol=0.20,
        )


def test_read_defaultvalues_duplicate_error(tmpdir, monkeypatch):
    """Test that read_defaultvalues raises ValueError for duplicate parameter names."""
    monkeypatch.chdir(tmpdir)

    # Create a simple Excel file with duplicate parameter names in defaultvalues
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

    defaultvalues.to_excel(
        "test_defaults.xlsx", sheet_name="defaultvalues", index=False
    )

    # Test that ValueError is raised with the exact expected message
    with pytest.raises(
        ValueError,
        match=(
            "Duplicate parameter names found in sheet "
            r"'defaultvalues': a, c\. All parameter names must be unique\."
        ),
    ):
        _read_defaultvalues("test_defaults.xlsx", "defaultvalues")


def _write_correlation_excel(filepath, names, lower_values):
    """Helper: write a correlation sheet with given lower-triangular values."""
    n = len(names)
    arr = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1):
            arr[i, j] = lower_values[i][j]
    df = pd.DataFrame(arr, index=names, columns=names)
    with pd.ExcelWriter(filepath, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="corr1")


def test_read_correlations_returns_symmetric_matrix(tmp_path):
    names = ["A", "B", "C"]
    lower = [[1.0], [0.5, 1.0], [0.3, 0.4, 1.0]]
    filepath = tmp_path / "corr.xlsx"
    _write_correlation_excel(filepath, names, lower)

    result = design_dist.read_correlations(str(filepath), corr_sheet="corr1")
    arr = result.to_numpy()

    np.testing.assert_array_almost_equal(arr, arr.T)
    np.testing.assert_array_almost_equal(np.diag(arr), [1.0, 1.0, 1.0])
    assert np.isclose(arr[1, 0], 0.5)
    assert np.isclose(arr[0, 1], 0.5)


def test_read_correlations_preserves_index_and_columns(tmp_path):
    names = ["X", "Y"]
    lower = [[1.0], [0.8, 1.0]]
    filepath = tmp_path / "corr.xlsx"
    _write_correlation_excel(filepath, names, lower)

    result = design_dist.read_correlations(str(filepath), corr_sheet="corr1")

    assert list(result.index) == names
    assert list(result.columns) == names


def test_read_correlations_to_numpy_is_writable(tmp_path):
    """The returned DataFrame's to_numpy(copy=True) should be writable."""
    names = ["A", "B"]
    lower = [[1.0], [0.5, 1.0]]
    filepath = tmp_path / "corr.xlsx"
    _write_correlation_excel(filepath, names, lower)

    result = design_dist.read_correlations(str(filepath), corr_sheet="corr1")
    arr = result.to_numpy(copy=True)
    arr[0, 1] = 0.99  # Should not raise


def test_print_corrmat_handles_negative_zeros(capsys):
    values = np.array([[1, -0.0, 0.9], [-0.0, 1, 0], [0.9, 0, 1.0]])
    df = pd.DataFrame(values, index=["A", "B", "C"], columns=["A", "B", "C"])
    print_corrmat(df)
    output = capsys.readouterr().out
    assert "1.00" in output
    assert ".90" in output


def test_print_corrmat_does_not_mutate_input():
    values = np.array([[1, -0.0, 0.9], [-0.0, 1, 0], [0.9, 0, 1.0]])
    df = pd.DataFrame(values, index=["A", "B", "C"], columns=["A", "B", "C"])
    original = df.copy()
    print_corrmat(df)
    pd.testing.assert_frame_equal(df, original)


def test_fill_with_background_values_no_index_column():
    dm = DesignMatrix()
    dm.designvalues = pd.DataFrame(
        {
            "SENSNAME": ["s1", "s1"],
            "SENSCASE": ["c1", "c1"],
            "param1": [np.nan, np.nan],
        }
    )
    dm.backgroundvalues = pd.DataFrame({"param1": [1.0, 2.0]})
    dm._fill_with_background_values()

    assert "index" not in dm.designvalues.columns


def test_fill_with_background_values_are_filled():
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

    assert dm.designvalues["param1"].tolist() == [10.0, 20.0]


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "-l"])
