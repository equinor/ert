"""Example use cases for fmudesign."""

import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from fmudesign import DesignMatrix, excel_to_dict
from fmudesign.fmudesignrunner import EXAMPLES

EXAMPLE_FILES = [example.filename for example in EXAMPLES]


def _run_cli(*args):
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.run(
        ["fmudesign", *map(str, args)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def test_prediction_rejection_sampled_ensemble(tmp_path):
    general_input = pd.DataFrame(
        data=[
            ["designtype", "onebyone"],
            ["repeats", 3],
            ["rms_seeds", "default"],
            ["background", "hmrealizations.xlsx"],
            ["distribution_seed", 42],
        ]
    )
    defaultvalues = pd.DataFrame(
        columns=["param_name", "default_value"],
        data=[
            ["HMREAL", "-1"],
            ["ORAT", 6000],
            ["RESTARTPATH", "FOO"],
            ["HMITER", "-1"],
        ],
    )
    pd.DataFrame(
        columns=["RESTARTPATH", "HMREAL", "HMITER"],
        data=[
            ["/scratch/foo/2020a_hm3/", 31, 3],
            ["/scratch/foo/2020a_hm3/", 38, 3],
            ["/scratch/foo/2020a_hm3/", 54, 3],
        ],
    ).to_excel(tmp_path / "hmrealizations.xlsx")

    input_path = tmp_path / "designinput.xlsx"
    with pd.ExcelWriter(input_path, engine="openpyxl") as writer:
        general_input.to_excel(
            writer, sheet_name="general_input", index=False, header=None
        )
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

    design = DesignMatrix()
    design.generate(excel_to_dict(input_path))

    assert set(design.designvalues["RESTARTPATH"]) == {"/scratch/foo/2020a_hm3/"}
    assert set(design.designvalues["HMITER"]) == {3}
    assert design.designvalues["REAL"].tolist() == list(range(6))
    assert design.designvalues["SENSNAME"].tolist() == ["ref"] * 3 + ["oil_rate"] * 3
    assert design.designvalues["HMREAL"].tolist() == [31, 38, 54] * 2


@pytest.mark.parametrize(
    "gen_input_sheet", ["general_input", "General_Input", "GENERALINPUT"]
)
def test_constant_distribution(tmp_path, gen_input_sheet):
    general_input = pd.DataFrame(
        data=[
            ["designtype", "onebyone"],
            ["repeats", 1],
            ["rms_seeds", "default"],
            ["distribution_seed", 42],
        ]
    )
    defaultvalues = pd.DataFrame(
        columns=["param_name", "default_value"], data=[["a", 1.0]]
    )
    design_input = pd.DataFrame(
        columns=[
            "sensname",
            "numreal",
            "type",
            "param_name",
            "dist_name",
            "dist_param1",
        ],
        data=[["montecarlo", 100, "dist", "a", "const", 1.0]],
    )

    input_path = tmp_path / "designinput.xlsx"
    with pd.ExcelWriter(input_path, engine="openpyxl") as writer:
        general_input.to_excel(
            writer, sheet_name=gen_input_sheet, index=False, header=None
        )
        design_input.to_excel(writer, sheet_name="designinput", index=False)
        defaultvalues.to_excel(writer, sheet_name="defaultvalues", index=False)

    design = DesignMatrix()
    design.generate(excel_to_dict(input_path, gen_input_sheet="generalinput"))

    assert len(design.designvalues) == 100
    assert set(design.designvalues["a"]) == {1.0}
    assert set(design.designvalues["SENSNAME"]) == {"montecarlo"}


@pytest.mark.slow
@pytest.mark.parametrize("verbosity", [1, 2])
def test_cli_verbosity_levels(tmp_path, monkeypatch, verbosity):
    monkeypatch.chdir(tmp_path)
    designfile = "ex4_background_parameters.xlsx"
    _run_cli("init", designfile)
    result = _run_cli("run", designfile, *(["--verbose"] * verbosity))

    assert (tmp_path / "generateddesignmatrix.xlsx").is_file()
    assert "CONTINUOUS PARAMETERS" in result.stdout
    assert "CORRELATION_GROUP 'corr1'" in result.stdout
    assert (tmp_path / "generateddesignmatrix/background/PARAM17.png").is_file()
    sensitivity_plot = tmp_path / "generateddesignmatrix/sens7/PARAM9.png"
    if verbosity == 2:
        assert sensitivity_plot.is_file()
    else:
        assert not sensitivity_plot.exists()


@pytest.mark.slow
def test_advanced_example_excel_parsing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _run_cli("init", "ex2_correlations.xlsx")
    _run_cli("init", "ex8_mc_with_correls.xlsx")

    correlations = excel_to_dict("ex2_correlations.xlsx")
    assert Path(correlations["background"]["extern"]).is_file()
    assert Path(correlations["sensitivities"]["sens8"]["extern_file"]).is_file()
    assert correlations["sensitivities"]["sens7"]["correlations"]["sheetnames"] == [
        "corr1"
    ]
    assert correlations["sensitivities"]["contacts"]["cases"] == {
        "shallow": {"PARAM2": -1, "PARAM3": -1, "PARAM4": -1},
        "deep": {"PARAM2": 1.0, "PARAM3": 1.0, "PARAM4": 1.0},
    }

    monte_carlo = excel_to_dict("ex8_mc_with_correls.xlsx")["sensitivities"][
        "montecarlo"
    ]
    assert monte_carlo["correlations"]["sheetnames"] == [
        "corr1",
        "corr2",
        "corr3",
    ]
    assert monte_carlo["dependencies"] == {
        "DATO": {
            "from_values": ["2018-11-02", "2018-11-03", "2018-11-04"],
            "to_params": {
                "DERIVED_PARAM1": ["1", "2", "3"],
                "DERIVED_PARAM2": ["a", "b", "c"],
            },
        }
    }


@pytest.mark.slow
@pytest.mark.parametrize("designfile", EXAMPLE_FILES, ids=EXAMPLE_FILES)
def test_all_example_files_cmd_init(tmp_path, monkeypatch, designfile):
    monkeypatch.chdir(tmp_path)
    _run_cli("init", designfile)
    _run_cli("run", designfile)

    design_values = pd.read_excel("generateddesignmatrix.xlsx", engine="openpyxl")
    assert not design_values.empty
    assert design_values.columns[:3].tolist() == ["REAL", "SENSNAME", "SENSCASE"]
    assert design_values["REAL"].tolist() == list(range(len(design_values)))
    assert not design_values.isna().any().any()
