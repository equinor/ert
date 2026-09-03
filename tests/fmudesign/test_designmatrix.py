"""Tests for the DesignMatrix API and installed command."""

import subprocess

import pandas as pd
import pytest

from fmudesign import DesignMatrix


def assert_valid_designmatrix(design_values):
    assert design_values.columns[:3].tolist() == ["REAL", "SENSNAME", "SENSCASE"]
    assert design_values["REAL"].tolist() == list(range(len(design_values)))
    assert not design_values.isna().any().any()


def test_designmatrix():
    design = DesignMatrix()
    design.generate(
        {
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
    )

    assert_valid_designmatrix(design.designvalues)
    assert len(design.designvalues) == 10
    assert isinstance(design.defaultvalues, dict)


@pytest.mark.slow
def test_endpoint_with_relative_input_and_custom_output_paths(tmp_path, monkeypatch):
    case_dir = tmp_path / "path" / "going" / "down"
    case_dir.mkdir(parents=True)
    subprocess.run(
        ["fmudesign", "init", "ex2_correlations.xlsx"],
        cwd=case_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    monkeypatch.chdir(tmp_path)

    source_design = case_dir / "ex2_correlations.xlsx"
    relative_design = case_dir.relative_to(tmp_path) / source_design.name
    output_path = tmp_path / "custom-design.xlsx"
    result = subprocess.run(
        ["fmudesign", str(relative_design), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Reading file:" in result.stdout
    assert "Reading background values from:" in result.stdout
    assert "Adjusted to nearest consistent correlation matrix:" in result.stdout
    assert "Design matrix of shape (91, 22) written to:" in result.stdout
    assert "Thank you for using fmudesign" in result.stdout

    assert output_path.is_file()
    assert_valid_designmatrix(pd.read_excel(output_path, engine="openpyxl"))


@pytest.mark.slow
def test_endpoint_resolves_external_seeds_file_relative_to_input(tmp_path, monkeypatch):
    """'rms_seeds' can also point to an external file. Like 'background' above,
    it must be resolved relative to the input file, not the CWD: the seeds file
    only exists in the nested case_dir, so a CWD-relative fallback would fail
    to find it.
    """
    case_dir = tmp_path / "path" / "going" / "down"
    case_dir.mkdir(parents=True)
    source_design = case_dir / "design-input.xlsx"
    with pd.ExcelWriter(source_design, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                ["designtype", "onebyone"],
                ["repeats", 3],
                ["rms_seeds", "seeds.xlsx"],
                ["background", "None"],
                ["distribution_seed", 42],
            ]
        ).to_excel(
            writer,
            sheet_name="general_input",
            index=False,
            header=False,
        )
        pd.DataFrame(
            [["rms_seed", None, "seed", None]],
            columns=["sensname", "numreal", "type", "param_name"],
        ).to_excel(writer, sheet_name="designinput", index=False)
        pd.DataFrame(
            [["RMS_SEED", 1000]],
            columns=["param_name", "default_value"],
        ).to_excel(writer, sheet_name="defaultvalues", index=False)
    pd.DataFrame([2000, 2001, 2002]).to_excel(
        case_dir / "seeds.xlsx",
        index=False,
        header=False,
    )
    monkeypatch.chdir(tmp_path)

    relative_design = case_dir.relative_to(tmp_path) / source_design.name
    output_path = tmp_path / "extseeds-design.xlsx"
    subprocess.run(
        ["fmudesign", str(relative_design), str(output_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    design_values = pd.read_excel(output_path, engine="openpyxl")
    assert_valid_designmatrix(design_values)
    # seeds.xlsx starts at 2000, unlike the 'default' 1000... sequence, so this
    # confirms the external file was actually read.
    assert design_values["RMS_SEED"].iloc[0] == 2000
