"""Testing generating design matrices from dictionary input"""

import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from semeio.fmudesign import DesignMatrix

TESTDATA = Path(__file__).parent / "data"


def matches(pattern: str, text: str) -> bool:
    """Match text against a pattern where <ANY> acts as a wildcard.

    Examples
    --------
    >>> matches("my name is <ANY>!", "my name is John!")
    True
    >>> matches("my <ANY> is <ANY>!", "my name is John!")
    True
    matches("my <ANY> is <ANY>!", "my name are John!")
    False
    """
    regex_pattern = re.escape(pattern)
    regex_pattern = regex_pattern.replace("<ANY>", ".+?")
    regex_pattern = f"^{regex_pattern}$"
    return bool(re.match(regex_pattern, text))


def valid_designmatrix(dframe):
    """Performs general checks on a design matrix, that should always be valid"""
    assert "REAL" in dframe

    # REAL always starts at 0 and is consecutive
    assert dframe["REAL"][0] == 0
    assert dframe["REAL"].diff().dropna().unique() == 1

    assert "SENSNAME" in dframe.columns
    assert "SENSCASE" in dframe.columns

    # There should be no empty cells in the dataframe:
    assert not dframe.isna().sum().sum()


def test_designmatrix():
    """Test the DesignMatrix class"""

    design = DesignMatrix()

    mock_dict = {
        "designtype": "onebyone",
        "seeds": "default",
        "repeats": 10,
        "distribution_seed": 42,
        "defaultvalues": {},
        "sensitivities": {
            "rms_seed": {
                "seedname": "RMS_SEED",
                "senstype": "seed",
                "parameters": None,
                "dependencies": {},
            }
        },
    }

    design.generate(mock_dict)
    valid_designmatrix(design.designvalues)
    assert len(design.designvalues) == 10
    assert isinstance(design.defaultvalues, dict)


def test_endpoint(tmpdir, monkeypatch):
    """Test the installed endpoint

    Will write generated design matrices to the pytest tmpdir directory,
    usually /tmp/pytest-of-<username>/
    """
    designfile = TESTDATA / "config/design_input_onebyone.xlsx"

    # The xlsx file contains a relative path, relative to the input design sheet:
    dependency = (
        pd.read_excel(designfile, header=None, engine="openpyxl")
        .set_index([0])[1]
        .to_dict()["background"]
    )

    tmpdir.chdir()
    monkeypatch.chdir(tmpdir)
    # Copy over input files:
    shutil.copy(str(designfile), ".")
    shutil.copy(Path(designfile).parent / dependency, ".")

    result = subprocess.run(
        ["fmudesign", str(designfile)], check=True, capture_output=True, text=True
    )

    # Use <ANY> in the string below to match anything in CLI output
    expected_output = """Reading file: <ANY>design_input_onebyone.xlsx'
    Reading background values from: <ANY>doe1.xlsx
     Generating sensitivity : seed
     Generating sensitivity : faults
     Generating sensitivity : velmodel
     Generating sensitivity : contacts
     Generating sensitivity : multz
     Generating sensitivity : sens6
     Generating sensitivity : sens7
    Sampling 4 parameters in correlation group 'corr1'

    Warning: Correlation matrix 'corr1' is inconsistent
    Requirements:
      - All diagonal elements must be 1
      - All elements must be between -1 and 1
      - The matrix must be positive semi-definite

    Input correlation matrix:
    |             |   (1) |   (2) |   (3) |   (4) |
    |:------------|------:|------:|------:|------:|
    | (1) PARAM9  |  1.00 |       |       |       |
    | (2) PARAM10 |  0.90 |  1.00 |       |       |
    | (3) PARAM11 |  0.00 |  0.90 |  1.00 |       |
    | (4) PARAM12 |  0.00 |  0.00 |  0.00 |  1.00 |

    Adjusted to nearest consistent correlation matrix:
    |             |   (1) |   (2) |   (3) |   (4) |
    |:------------|------:|------:|------:|------:|
    | (1) PARAM9  |  1.00 |       |       |       |
    | (2) PARAM10 |  0.74 |  1.00 |       |       |
    | (3) PARAM11 |  0.11 |  0.74 |  1.00 |       |
    | (4) PARAM12 |  0.00 |  0.00 |  0.00 |  1.00 |
    Generating sensitivity : sens8
Provided number of background values (11) is smaller than number of realisations for sensitivity ('sens7', 'p10_p90') and parameter PARAM13. Will be filled with default values.
Provided number of background values (11) is smaller than number of realisations for sensitivity ('sens7', 'p10_p90') and parameter PARAM14. Will be filled with default values.
Provided number of background values (11) is smaller than number of realisations for sensitivity ('sens7', 'p10_p90') and parameter PARAM15. Will be filled with default values.
Provided number of background values (11) is smaller than number of realisations for sensitivity ('sens7', 'p10_p90') and parameter PARAM16. Will be filled with default values.
Design matrix of shape (91, 22) written to: 'generateddesignmatrix.xlsx'

 Thank you for using fmudesign <ANY>
  - Documentation:           https://equinor.github.io/fmu-tools/fmudesign.html
  - Course docs:             https://fmu-docs.equinor.com/docs/fmu-coursedocs/fmu-howto/sensitivities/index.html
  - Issues/feature requests: https://github.com/equinor/semeio/issues"""  # noqa: E501

    for stdout_line, expected_line in zip(
        result.stdout.split(), expected_output.split(), strict=False
    ):
        assert matches(expected_line, stdout_line)

    assert Path("generateddesignmatrix.xlsx").exists  # Default output file
    valid_designmatrix(pd.read_excel("generateddesignmatrix.xlsx", engine="openpyxl"))

    subprocess.run(["fmudesign", str(designfile), "anotheroutput.xlsx"], check=True)
    assert Path("anotheroutput.xlsx").exists
