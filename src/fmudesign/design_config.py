from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt

from ert.config.design_matrix import read_default_values
from fmudesign.config_validation import SeedStrategy
from fmudesign.design_input import DesignInput
from fmudesign.general_input import GeneralInput
from fmudesign.read_background import read_background
from fmudesign.utils import seeds_from_extern


class DesignConfig(BaseModel):
    input_file: str
    designtype: Literal["onebyone"]
    repeats: PositiveInt
    distribution_seed: NonNegativeInt | None
    background: dict[str, Any] | None
    seeds: Literal["default"] | Annotated[list[int], Field(min_length=1)] | None
    correlation_iterations: NonNegativeInt
    seed_strategy: SeedStrategy
    defaultvalues: dict[str, str | float | int | bool]
    sensitivities: dict[str, Any]
    decimals: dict[str, int] | None

    @classmethod
    def from_input(
        cls,
        input_filename: str,
        general_input: GeneralInput,
        design_input: DesignInput,
        default_values_sheet: str,
    ) -> Self:
        """Reads configuration from Excel file for a onebyone design matrix.

        Args:
            input_filename (str): Name of excel workbook
            general_input (GeneralInput): Validated general input
            design_input (DesignInput): Validated design input
            default_values_sheet (str): name of default value sheet

        Returns:
            Config for how to generate a design matrix
        """
        if isinstance(seeds := general_input.rms_seeds, Path):
            rms_seeds: Literal["default"] | list[int] | None = seeds_from_extern(seeds)
            if not rms_seeds:
                raise ValueError(
                    f"Empty rms_seeds file '{seeds.resolve()}', "
                    f"must contain at least one seed."
                )
        else:
            rms_seeds = seeds

        if isinstance(bgr := general_input.background, Path):
            background = {"extern": str(bgr)}
        elif isinstance(bgr, str):
            background = read_background(input_filename, bgr)
        else:
            background = None
        default_values = read_default_values(
            Path(input_filename), default_values_sheet, has_header=True
        )

        return cls(
            **(
                general_input.model_dump()
                | asdict(design_input)
                | {
                    "input_file": input_filename,
                    "background": background,
                    "seeds": rms_seeds,
                    "defaultvalues": default_values,
                }
            )
        )
