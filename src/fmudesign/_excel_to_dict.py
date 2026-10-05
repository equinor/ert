"""This module contains functions for reading Excel config files.
These are converted to a dict-of-dicts representation, then they are used
by the DesignMatrix class to generate design matrices.
"""

from pathlib import Path
from typing import Any, Literal

import openpyxl
import yaml

from ert.config.design_matrix import read_default_values

from .design_input import DesignInput
from .general_input import GeneralInput
from .read_background import read_background
from .utils import (
    excel_sheet_names,
    find_sheet,
    seeds_from_extern,
)


def excel_to_dict(
    input_filename: str,
    *,
    gen_input_sheet: str = "general_input",
    design_input_sheet: str = "designinput",
    default_val_sheet: str = "defaultvalues",
) -> dict[str, Any]:
    """Read excel file with input to design setup

    Args:
        input_filename (str): Name of excel input file
        gen_input_sheet (str): Sheet name for general input
        design_input_sheet (str): Sheet name for design input
        default_val_sheet (str): Sheet name for default input

    Returns:
        dict on format for DesignMatrix
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

    return _excel_to_dict_onebyone(
        input_filename=input_filename,
        general_input=general_input,
        design_input=design_input,
        default_values_sheet=default_values_sheet,
    )


def inputdict_to_yaml(inputdict: dict[str, Any], filename: str) -> None:
    """Write inputdict to yaml format

    Args:
        inputdict (dict)
        filename (str): name of output file
    """
    with Path(filename).open("w", encoding="utf-8") as stream:
        yaml.dump(inputdict, stream)


def _excel_to_dict_onebyone(
    input_filename: str,
    *,
    general_input: GeneralInput,
    design_input: DesignInput,
    default_values_sheet: str,
) -> dict[str, Any]:
    """Reads configuration from Excel file for a onebyone design matrix.

    Args:
        input_filename (str): Name of excel workbook
        general_input (GeneralInput): Validated general input
        design_input (DesignInput): Validated design input
        default_values_sheet (str): name of default value sheet

    Returns:
        dict on format for DesignMatrix
    """

    if isinstance(seeds := general_input.rms_seeds, Path):
        rms_seeds: Literal["default"] | list[int] | None = seeds_from_extern(seeds)
    else:
        rms_seeds = seeds

    if isinstance(bgr := general_input.background, Path):
        background = {"extern": str(bgr)}
    elif isinstance(bgr, str):
        background = read_background(input_filename, bgr)
    else:
        background = None

    output: dict[str, Any] = {
        "input_file": input_filename,
        "designtype": general_input.designtype,
        "repeats": general_input.repeats,
        "distribution_seed": general_input.distribution_seed,
        "background": background,
        "seeds": rms_seeds,
        "correlation_iterations": general_input.correlation_iterations,
        "seed_strategy": general_input.seed_strategy,
        "defaultvalues": read_default_values(
            Path(input_filename), default_values_sheet, has_header=True
        ),
        "sensitivities": design_input.sensitivities,
    }  # This is the config that we read and return

    if design_input.decimals is not None:
        output["decimals"] = design_input.decimals
    return output


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
