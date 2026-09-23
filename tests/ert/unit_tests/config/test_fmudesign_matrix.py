import shutil
import sys
from importlib.resources import as_file, files
from pathlib import Path

import pandas as pd
import pytest
from polars.testing import assert_frame_equal

import fmudesign
from ert.config import ConfigValidationError, DesignMatrix, ErtConfig, FmuDesignMatrix
from fmudesign import fmudesignrunner

SEEDED_EXAMPLES = [
    "fmudesign_ex_montecarlo.xlsx",
    "ex1_onebyone_rms_repeat.xlsx",
    "ex2_correlations.xlsx",
    "ex5_single_reference.xlsx",
    "ex6_singlereference_and_seed.xlsx",
    "ex8_mc_with_correls.xlsx",
]
DESIGN_INPUT_COLUMNS = [
    "sensname",
    "numreal",
    "type",
    "senscase1",
    "senscase2",
    "param_name",
    "value1",
    "value2",
    "dist_name",
    "dist_param1",
    "dist_param2",
]
OIL_RATE_SENSITIVITY = {
    "sensname": "oil_rate",
    "type": "dist",
    "param_name": "ORAT",
    "dist_name": "uniform",
    "dist_param1": 5000,
    "dist_param2": 9000,
}


def _copy_example(filename: str, destination: Path) -> Path:
    example = next(ex for ex in fmudesignrunner.EXAMPLES if ex.filename == filename)
    for name in [example.filename, *example.other_files]:
        with as_file(files("fmudesign.examples") / name) as source:
            shutil.copy(source, destination / name)
    return destination / filename


def _write_fmudesign_input(
    path: Path,
    sensitivities: list[dict[str, object]],
    defaults: dict[str, object],
    distribution_seed: int | None = 42,
    sheet_names: tuple[str, str, str] = (
        "general_input",
        "designinput",
        "defaultvalues",
    ),
) -> None:
    general_input_sheet, design_input_sheet, default_values_sheet = sheet_names
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                ["designtype", "onebyone"],
                ["repeats", 2],
                ["rms_seeds", "None"],
                [
                    "distribution_seed",
                    "None" if distribution_seed is None else distribution_seed,
                ],
            ]
        ).to_excel(writer, sheet_name=general_input_sheet, index=False, header=False)
        pd.DataFrame(sensitivities, columns=DESIGN_INPUT_COLUMNS).to_excel(
            writer, sheet_name=design_input_sheet, index=False
        )
        pd.DataFrame(
            list(defaults.items()), columns=["param_name", "default_value"]
        ).to_excel(writer, sheet_name=default_values_sheet, index=False)


def _assert_same_as_generated_file(input_file: Path, tmp_path: Path) -> None:
    design = fmudesign.DesignMatrix()
    design.generate(fmudesign.excel_to_dict(str(input_file)))
    design.to_xlsx(str(tmp_path / "generated.xlsx"))
    from_generated_file = DesignMatrix(
        tmp_path / "generated.xlsx", "DesignSheet01", "DefaultValues"
    )

    from_input_file = FmuDesignMatrix(input_file)

    assert_frame_equal(
        from_input_file.design_matrix_df, from_generated_file.design_matrix_df
    )
    assert (
        from_input_file.parameter_configurations
        == from_generated_file.parameter_configurations
    )
    assert (
        from_input_file.active_realizations == from_generated_file.active_realizations
    )


@pytest.mark.parametrize("example", SEEDED_EXAMPLES)
def test_that_fmudesign_input_gives_the_same_design_matrix_as_its_generated_file(
    example, tmp_path
):
    _assert_same_as_generated_file(_copy_example(example, tmp_path), tmp_path)


def test_that_booleans_among_text_match_the_generated_file(tmp_path):
    input_file = tmp_path / "input.xlsx"
    _write_fmudesign_input(
        input_file,
        [
            {"sensname": "ref", "type": "ref"},
            {
                "sensname": "flag",
                "type": "scenario",
                "senscase1": "on",
                "param_name": "FLAG",
                "value1": "category",
            },
        ],
        defaults={"FLAG": True},
    )

    _assert_same_as_generated_file(input_file, tmp_path)


def test_that_fmudesign_input_requires_a_distribution_seed(tmp_path):
    input_file = tmp_path / "input.xlsx"
    _write_fmudesign_input(
        input_file, [OIL_RATE_SENSITIVITY], {"ORAT": 6000}, distribution_seed=None
    )

    with pytest.raises(ConfigValidationError, match="distribution_seed"):
        FmuDesignMatrix(input_file)


def test_that_corrupt_fmudesign_input_raises_config_validation_error(tmp_path):
    input_file = tmp_path / "input.xlsx"
    input_file.write_bytes(b"not an Excel file")

    with pytest.raises(ConfigValidationError, match="could not be loaded"):
        FmuDesignMatrix(input_file)


@pytest.mark.usefixtures("use_tmpdir")
def test_that_fmudesign_keyword_generates_design_parameters_from_the_given_sheets():
    _write_fmudesign_input(
        Path("input.xlsx"),
        [OIL_RATE_SENSITIVITY],
        {"ORAT": 6000},
        sheet_names=("general", "design", "defaults"),
    )

    config = ErtConfig.from_file_contents(
        "NUM_REALIZATIONS 2\n"
        "FMUDESIGN input.xlsx GENERAL_INPUT_SHEET:general "
        "DESIGN_INPUT_SHEET:design DEFAULT_VALUES_SHEET:defaults"
    )

    assert "ORAT" in {
        parameter.name
        for parameter in config.parameter_configurations_with_design_matrix
    }


def test_that_fmudesign_keyword_gives_the_same_parameters_as_running_fmudesign(
    tmp_path, monkeypatch
):
    # Correlated distributions, scenarios, seeds, an external design and rounding
    (tmp_path / "input").mkdir()
    input_file = _copy_example("ex2_correlations.xlsx", tmp_path / "input")
    monkeypatch.setattr(
        sys,
        "argv",
        ["fmudesign", "run", str(input_file), str(tmp_path / "generated.xlsx")],
    )
    fmudesignrunner.main()
    (tmp_path / "via_fmudesign.ert").write_text(
        "NUM_REALIZATIONS 100\n"
        "DESIGN_MATRIX generated.xlsx DESIGN_SHEET:DesignSheet01 "
        "DEFAULT_SHEET:DefaultValues\n"
    )
    (tmp_path / "via_keyword.ert").write_text(
        "NUM_REALIZATIONS 100\nFMUDESIGN input/ex2_correlations.xlsx\n"
    )

    via_fmudesign = ErtConfig.from_file(str(tmp_path / "via_fmudesign.ert"))
    via_keyword = ErtConfig.from_file(str(tmp_path / "via_keyword.ert"))

    assert_frame_equal(
        via_keyword.analysis_config.design_matrix.design_matrix_df,
        via_fmudesign.analysis_config.design_matrix.design_matrix_df,
    )
    assert (
        via_keyword.parameter_configurations_with_design_matrix
        == via_fmudesign.parameter_configurations_with_design_matrix
    )
    assert via_keyword.active_realizations == via_fmudesign.active_realizations
