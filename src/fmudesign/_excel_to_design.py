"""This module contains functions for reading Excel config files.
These are converted to a dict-of-dicts representation, then they are used
by the DesignMatrix class to generate design matrices.
"""

from pathlib import Path

import openpyxl
import yaml

from .design_config import DesignConfig
from .design_input import DesignInput
from .general_input import GeneralInput
from .utils import (
    excel_sheet_names,
    find_sheet,
)


def excel_to_config(
    input_filename: str,
    *,
    gen_input_sheet: str = "general_input",
    design_input_sheet: str = "designinput",
    default_val_sheet: str = "defaultvalues",
) -> DesignConfig:
    """Read excel file with input to design configuration

    Args:
        input_filename (str): Name of excel input file
        gen_input_sheet (str): Sheet name for general input
        design_input_sheet (str): Sheet name for design input
        default_val_sheet (str): Sheet name for default input

    Returns:
        DesignConfig for DesignMatrix
    """
    # To be backwards compatible, we do not change the input arg names
    general_input_sheet = gen_input_sheet
    default_values_sheet = default_val_sheet

    # Find sheets
    _assert_no_merged_cells(input_filename)
    sheet_names = excel_sheet_names(input_filename)
    general_input_sheet = find_sheet(general_input_sheet, names=sheet_names)
    design_input_sheet = find_sheet(design_input_sheet, names=sheet_names)
    default_values_sheet = find_sheet(default_values_sheet, names=sheet_names)

    general_input = GeneralInput.from_xlsx(input_filename, general_input_sheet)
    design_input = DesignInput.from_xlsx(input_filename, design_input_sheet)

    return DesignConfig.from_input(
        input_filename=input_filename,
        general_input=general_input,
        design_input=design_input,
        default_values_sheet=default_values_sheet,
    )


def config_to_yaml(config: DesignConfig, filename: str) -> None:
    """Write inputdict to yaml format

    Args:
        config (DesignConfig)
        filename (str): name of output file
    """
    with Path(filename).open("w", encoding="utf-8") as stream:
        yaml.dump(config.model_dump(mode="json"), stream, encoding="utf-8")


def _assert_no_merged_cells(input_filename: str) -> None:
    """Raises an exception if any merged cells exist, else returns None."""

    workbook = openpyxl.load_workbook(input_filename)
    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        merged_ranges = list(worksheet.merged_cells.ranges)
        if merged_ranges:
            raise ValueError(
                "Merged cells are not allowed. Found merged cell in "
                f"{input_filename} at sheet '{sheet_name}'.\n"
                f"Found {len(merged_ranges)} merged cell range(s): {merged_ranges}"
            )
