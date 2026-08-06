"""Example use cases for semeio.fmudesign"""

import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from semeio.fmudesign import DesignMatrix, excel_to_dict
from semeio.fmudesign.fmudesignrunner import EXAMPLES

EXAMPLE_FILES = [ex.filename for ex in EXAMPLES]

TESTDATA = Path(__file__).parent / "data"
TEST_FILES = list((TESTDATA / "config").glob("design_input*.xlsx"))


def test_prediction_rejection_sampled_ensemble(tmpdir, monkeypatch):
    """Test making a design matrix for prediction realizations based on an
    ensemble made with manual history matching (rejection sampling).

    In the use-case this test is modelled on, the design matrix is used
    to set up a prediction ensemble where each DATA file points to another
    Eclipse run on disk which contains the history, identified by the
    realization index ("HMREAL") in the history match run.
    """
    monkeypatch.chdir(tmpdir)
    general_input = pd.DataFrame(
        data=[
            ["designtype", "onebyone"],
            ["repeats", 3],  # This matches the number of HM-samples we have.
            ["rms_seeds", "default"],  # Geogrid from HM realization is used
            ["background", "hmrealizations.xlsx"],
            ["distribution_seed", 42],
        ]
    )
    defaultvalues = pd.DataFrame(
        columns=["param_name", "default_value"],
        data=[
            # All background parameters must be mentioned in
            # DefaultValues (but these defaults are not used in
            # this particular test scenario)
            ["HMREAL", "-1"],
            ["ORAT", 6000],
            ["RESTARTPATH", "FOO"],
            ["HMITER", "-1"],
        ],
    )

    # Background to separate file, these define some history realizations that
    # all scenarios should run over:
    pd.DataFrame(
        columns=["RESTARTPATH", "HMREAL", "HMITER"],
        data=[
            ["/scratch/foo/2020a_hm3/", 31, 3],
            ["/scratch/foo/2020a_hm3/", 38, 3],
            ["/scratch/foo/2020a_hm3/", 54, 3],
        ],
    ).to_excel("hmrealizations.xlsx")

    writer = pd.ExcelWriter("designinput.xlsx", engine="openpyxl")
    general_input.to_excel(writer, sheet_name="general_input", index=False, header=None)
    pd.DataFrame(
        columns=[
            "sensname",
            "numreal",
            "type",
            "param_name",
            "dist_name",
            "dist_param1",
            "dist_param2",
        ],
        data=[
            ["ref", None, "background", None],
            ["oil_rate", None, "dist", "ORAT", "uniform", 5000, 9000],
        ],
    ).to_excel(writer, sheet_name="design_input", index=False)
    defaultvalues.to_excel(writer, sheet_name="defaultvalues", index=False)
    writer.close()

    dict_design = excel_to_dict("designinput.xlsx")
    design = DesignMatrix()
    design.generate(dict_design)

    assert set(design.designvalues["RESTARTPATH"]) == {"/scratch/foo/2020a_hm3/"}
    assert set(design.designvalues["HMITER"]) == {3}
    assert all(design.designvalues["REAL"] == [0, 1, 2, 3, 4, 5])
    assert all(
        design.designvalues["SENSNAME"]
        == [
            "ref",
            "ref",
            "ref",
            "oil_rate",
            "oil_rate",
            "oil_rate",
        ]
    )

    # This is the most important bit in this test function, that the realization
    # list in the background xlsx is repeated for each sensitivity:
    assert all(design.designvalues["HMREAL"] == [31, 38, 54, 31, 38, 54])


@pytest.mark.parametrize(
    "gen_input_sheet", ["general_input", "General_Input", "GENERALINPUT"]
)
def test_constant_distribution(tmpdir, monkeypatch, gen_input_sheet):
    """Create a design matrix workbook with a single constant parameter 'a'."""
    monkeypatch.chdir(tmpdir)

    # General input configuration
    general_input = pd.DataFrame(
        data=[
            ["designtype", "onebyone"],
            ["repeats", 1],
            ["rms_seeds", "default"],
            ["distribution_seed", 42],
        ]
    )

    # Default values for parameters
    defaultvalues = pd.DataFrame(
        columns=["param_name", "default_value"],
        data=[
            ["a", 1.0],
        ],
    )

    # Design input with single constant parameter
    design_input = pd.DataFrame(
        columns=[
            "sensname",
            "numreal",
            "type",
            "param_name",
            "dist_name",
            "dist_param1",
        ],
        data=[
            ["montecarlo", 100, "dist", "a", "const", 1.0],
        ],
    )

    # Create Excel workbook with all sheets
    writer = pd.ExcelWriter("designinput.xlsx", engine="openpyxl")
    general_input.to_excel(writer, sheet_name=gen_input_sheet, index=False, header=None)
    design_input.to_excel(writer, sheet_name="designinput", index=False)
    defaultvalues.to_excel(writer, sheet_name="defaultvalues", index=False)
    writer.close()

    # Generate design matrix
    dict_design = excel_to_dict("designinput.xlsx", gen_input_sheet="generalinput")
    design = DesignMatrix()
    design.generate(dict_design)

    # Print results
    print(f"Parameter 'a' values: {set(design.designvalues['a'])}")
    print(f"Number of realizations: {len(design.designvalues)}")
    print(f"Sensitivity name: {set(design.designvalues['SENSNAME'])}")


@pytest.mark.parametrize("designfile", TEST_FILES, ids=[p.stem for p in TEST_FILES])
@pytest.mark.parametrize("verbosity", [0, 1, 2])
def test_all_input_files(tmpdir, monkeypatch, designfile, verbosity):
    """Smoketest all files."""

    monkeypatch.chdir(tmpdir)

    # Copy all example files over, to guarantee existence of dependency files
    for filename in designfile.parent.glob("*"):
        if Path(filename).is_file():
            shutil.copy(filename, ".")

    # Run the CLI tool (test will fail on non-zero status code)
    verbose = ["--verbose"] * verbosity
    subprocess.run(
        ["fmudesign", designfile, *verbose], check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize("designfile", EXAMPLE_FILES, ids=EXAMPLE_FILES)
@pytest.mark.parametrize("verbosity", [0])
def test_all_example_files_cmd_init(tmpdir, monkeypatch, designfile, verbosity):
    """Smoketest all files available in fmudesign init subcommand."""
    monkeypatch.chdir(tmpdir)
    subprocess.run(
        ["fmudesign", "init", designfile], check=True, capture_output=True, text=True
    )

    # Run the CLI tool (test will fail on non-zero status code)
    verbose = ["--verbose"] * verbosity
    subprocess.run(
        ["fmudesign", "run", designfile, *verbose],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("designfile", TEST_FILES, ids=[p.stem for p in TEST_FILES])
def test_all_input_files_relative_paths(tmpdir, monkeypatch, designfile):
    """Smoketest all files, but invoke them from a directory above.
    This tests that relative paths in the Excel files work correctly."""

    monkeypatch.chdir(tmpdir)
    copy_to = os.path.join(".", "path", "going", "down")

    Path(copy_to).mkdir(exist_ok=True, parents=True)

    # Copy all example files over, to guarantee existence of dependency files
    for filename in designfile.parent.glob("*"):
        if Path(filename).is_file():
            shutil.copy(filename, copy_to)

    # Run the CLI tool (test will fail on non-zero status code)
    subprocess.run(
        ["fmudesign", os.path.join(".", "path", "going", "down", designfile)],
        check=True,
        capture_output=True,
        text=True,
    )
