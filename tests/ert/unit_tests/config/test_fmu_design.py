from pathlib import Path
from textwrap import dedent

import openpyxl
import polars as pl
import pytest

from ert.config import ConfigValidationError, ErtConfig


def write_fmu_design_workbook(
    path: str | Path,
    *,
    design_sheet: str = "designinput",
    default_sheet: str = "defaultvalues",
    design_values: str = "faults,,scenario,FAULT_POSITION,east,-1,west,1",
    default_values: str = "FAULT_POSITION,0\nOWC,2650",
) -> None:
    sheets = {
        "general_input": dedent(
            """\
            designtype,onebyone
            repeats,1
            rms_seeds,None
            background,None
            distribution_seed,None"""
        ),
        design_sheet: (
            "sensname,numreal,type,param_name,senscase1,value1,senscase2,value2\n"
            f"{design_values}"
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
