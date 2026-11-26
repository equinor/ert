from datetime import date
from pathlib import Path
from textwrap import dedent

import openpyxl
import polars as pl
import pytest

from ert.config import ConfigValidationError, ConfigWarning, ErtConfig


def write_fmu_design_workbook(
    path: str | Path,
    *,
    design_sheet: str = "designinput",
    default_sheet: str = "defaultvalues",
    design_values: str = "faults,,scenario,FAULT_POSITION,east,-1,west,1",
    default_values: str = "FAULT_POSITION,0\nOWC,2650",
    rms_seeds: str = "None",
) -> None:
    sheets = {
        "general_input": dedent(
            f"""\
            designtype,onebyone
            repeats,1
            rms_seeds,{rms_seeds}
            background,None
            distribution_seed,None"""
        ),
        design_sheet: (
            "sensname,numreal,type,param_name,senscase1,value1,senscase2,value2,"
            f"dist_name,dist_param1,dist_param2\n{design_values}"
        ),
        default_sheet: f"param_name,default_values\n{default_values}",
    }

    workbook = openpyxl.Workbook(write_only=True)
    for sheet_name, csv in sheets.items():
        sheet = workbook.create_sheet(sheet_name)
        for line in csv.splitlines():
            sheet.append(line.split(","))
    workbook.save(path)


def parse_ert_config_with_fmu_design(fmu_design_line: str) -> ErtConfig:
    Path("config.ert").write_text(
        f"""
        NUM_REALIZATIONS 2
        {fmu_design_line}
        """,
        encoding="utf-8",
    )
    return ErtConfig.from_file("config.ert")


def test_that_a_fmu_design_config_workbook_is_parsed_to_design_matrix(use_tmpdir):
    design_file = "design.xlsx"
    write_fmu_design_workbook(design_file)

    ert_config = parse_ert_config_with_fmu_design("FMU_DESIGN design.xlsx")

    design_matrix = ert_config.analysis_config.design_matrix
    assert design_matrix is not None
    assert design_matrix.origin == f"{design_file} (designinput defaultvalues)"


def test_that_origin_sheet_names_default_to_fmudesigns_default_sheet_names(use_tmpdir):
    """This test is made to fail if the default sheet names of
    fmudesign.excel_to_config are changed, as ert is currently hardcoding these
    in analysis_config to avoid importing sheet names from fmudesign.
    """
    design_file = "design.xlsx"
    write_fmu_design_workbook(design_file)
    ert_config = parse_ert_config_with_fmu_design(f"FMU_DESIGN {design_file}")
    design_matrix = ert_config.analysis_config.design_matrix

    assert design_matrix.origin == "design.xlsx (designinput defaultvalues)"


def test_that_sheet_names_can_be_provided_to_fmu_design_config_keyword(use_tmpdir):
    design_file = "design.xlsx"
    custom_design_sheet = "foo.xlsx"
    custom_default_sheet = "bar.xlsx"
    write_fmu_design_workbook(
        design_file,
        design_sheet=custom_design_sheet,
        default_sheet=custom_default_sheet,
    )

    ert_config = parse_ert_config_with_fmu_design(
        f"FMU_DESIGN {design_file} "
        f"DESIGN_SHEET:{custom_design_sheet} DEFAULT_SHEET:{custom_default_sheet}"
    )

    design_matrix = ert_config.analysis_config.design_matrix
    assert (
        design_matrix.origin
        == f"{design_file} ({custom_design_sheet} {custom_default_sheet})"
    )


def test_that_missing_fmu_design_sheet_raises_config_validation_error(use_tmpdir):
    design_file = "design.xlsx"
    write_fmu_design_workbook(design_file)

    with pytest.raises(ConfigValidationError, match="nosuchsheet"):
        parse_ert_config_with_fmu_design(
            f"FMU_DESIGN {design_file} DESIGN_SHEET:nosuchsheet"
        )


def test_that_realization_row_in_fmu_design_raises_config_validation_error(
    use_tmpdir,
):
    design_file = "design.xlsx"
    write_fmu_design_workbook(
        design_file, default_values="FAULT_POSITION,0\nrealization,100"
    )

    with pytest.raises(ConfigValidationError, match="reserved internal keyword"):
        parse_ert_config_with_fmu_design(f"FMU_DESIGN {design_file}")


def test_that_mix_typed_colum_in_fmu_design_design_matrix_is_cast_to_str(
    use_tmpdir,
):
    design_file = "design.xlsx"
    write_fmu_design_workbook(
        design_file,
        design_values="faults,,scenario,FAULT_POSITION,east,-1,west,west",
    )

    ert_config = parse_ert_config_with_fmu_design(f"FMU_DESIGN {design_file}")

    design_matrix = ert_config.analysis_config.design_matrix

    fault_position = design_matrix.design_matrix_df.get_column("FAULT_POSITION")
    assert fault_position.dtype == pl.String
    assert fault_position.to_list() == ["-1", "west"]
    assert "FAULT_POSITION" in design_matrix.categorical_parameters


def test_that_non_existing_design_file_raises_config_validation_error(use_tmpdir):
    design_file = "design.xlsx"

    with pytest.raises(
        ConfigValidationError, match=f'Cannot find file or directory "{design_file}"'
    ):
        parse_ert_config_with_fmu_design(f"FMU_DESIGN {design_file}")


def test_that_empty_fmu_design_file_raises_config_validation_error(use_tmpdir):
    design_file = "design.xlsx"
    Path(design_file).touch()

    with pytest.raises(
        ConfigValidationError, match=f"{design_file}.* not a valid .xlsx file"
    ):
        parse_ert_config_with_fmu_design(f"FMU_DESIGN {design_file}")


def test_that_invalid_fmu_design_file_extension_raises_config_validation_error(
    use_tmpdir,
):
    design_file = "design.txt"
    Path(design_file).touch()

    with pytest.raises(
        ConfigValidationError, match=f"must have file extension .xlsx; .*{design_file}"
    ):
        parse_ert_config_with_fmu_design(f"FMU_DESIGN {design_file}")


def test_that_whole_number_columns_in_fmu_design_design_matrix_are_int64(
    use_tmpdir,
):
    design_file = "design.xlsx"
    write_fmu_design_workbook(
        design_file,
        design_values=(
            "faults,,scenario,FAULT_POSITION,east,-1,west,1\nreference,,ref"
        ),
    )

    ert_config = parse_ert_config_with_fmu_design(f"FMU_DESIGN {design_file}")

    fault_position = ert_config.analysis_config.design_matrix.design_matrix_df[
        "FAULT_POSITION"
    ]
    assert fault_position.dtype == pl.Int64
    assert fault_position.to_list() == [-1, 1, 0]


def test_that_date_columns_in_fmu_design_are_converted_to_strings_with_warning(
    use_tmpdir,
):
    design_file = "design.xlsx"
    write_fmu_design_workbook(design_file)
    workbook = openpyxl.load_workbook(design_file)
    workbook["designinput"]["F2"] = date(2024, 1, 1)
    workbook["designinput"]["H2"] = date(2025, 1, 1)
    workbook["defaultvalues"]["B2"] = date(2020, 1, 1)
    workbook.save(design_file)

    with pytest.warns(ConfigWarning, match="date/datetime columns"):
        ert_config = parse_ert_config_with_fmu_design(f"FMU_DESIGN {design_file}")

    fault_position = ert_config.analysis_config.design_matrix.design_matrix_df[
        "FAULT_POSITION"
    ]
    assert fault_position.dtype == pl.String


def test_that_multiword_parameter_name_in_fmu_design_raises_config_validation_error(
    use_tmpdir,
):
    design_file = "design.xlsx"
    multiword_name = "FAULT POSITION"
    write_fmu_design_workbook(
        design_file,
        design_values=f"faults,,scenario,{multiword_name},east,-1,west,1",
        default_values=f"{multiword_name},0",
    )

    with pytest.raises(
        ConfigValidationError,
        match=f"Multiple words in parameter name .*{multiword_name}",
    ):
        parse_ert_config_with_fmu_design(f"FMU_DESIGN {design_file}")


def test_that_generated_empty_cell_raises_config_validation_error(
    use_tmpdir,
):
    """fmudesign assigns no RMS_SEED to ref rows, so the cell is left empty
    unless RMS_SEED has a default value.
    """
    design_file = "design.xlsx"
    write_fmu_design_workbook(
        design_file,
        design_values="faults,,scenario,FAULT_POSITION,east,-1,west,1\nreference,,ref",
        rms_seeds="default",
    )

    with pytest.raises(ConfigValidationError, match="empty cells"):
        parse_ert_config_with_fmu_design(f"FMU_DESIGN {design_file}")


def test_that_fmu_design_samples_are_reproducible_with_ert_random_seed(use_tmpdir):
    design_file = "design.xlsx"
    write_fmu_design_workbook(
        design_file,
        design_values="perm,2,dist,PERM,,,,,normal,0,1",
        default_values="PERM,0",
    )

    dm1, dm2 = (
        parse_ert_config_with_fmu_design(
            f"FMU_DESIGN {design_file}\nRANDOM_SEED 123"
        ).analysis_config.design_matrix.design_matrix_df
        for _ in range(2)
    )

    assert dm1.equals(dm2)


def test_that_negative_random_seed_propagated_to_fmu_design_raises_config_error(
    use_tmpdir,
):
    negative_random_seed = -1
    design_file = "design.xlsx"
    write_fmu_design_workbook(design_file)

    with pytest.raises(
        ConfigValidationError,
        match=f"must be a non-negative integer, was '{negative_random_seed}'",
    ):
        parse_ert_config_with_fmu_design(
            f"FMU_DESIGN {design_file}\nRANDOM_SEED {negative_random_seed}"
        )


def test_that_fmudesign_output_is_silenced_when_run_from_ert(
    use_tmpdir, capsys, caplog
):
    design_file = "design.xlsx"
    write_fmu_design_workbook(design_file)

    parse_ert_config_with_fmu_design(f"FMU_DESIGN {design_file}")

    stdout = capsys.readouterr().out
    assert "Generating sensitivity" not in stdout
    assert "Setting to default" not in stdout
