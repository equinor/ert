from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from ert.config.design_matrix import read_default_values
from fmudesign.design_input import DesignInput
from fmudesign.general_input import GeneralInput
from fmudesign.read_background import read_background
from fmudesign.seed_strategy import SeedStrategy
from fmudesign.utils import seeds_from_extern


class DesignConfig(BaseModel):
    designtype: Literal["onebyone"]
    repeats: PositiveInt
    defaultvalues: dict[str, str | float | int | bool]
    sensitivities: dict[str, Any]
    correlation_iterations: NonNegativeInt = 0
    distribution_seed: NonNegativeInt | None = None
    background: dict[str, Any] | None = None
    seeds: Literal["default"] | Annotated[list[int], Field(min_length=1)] | None = None
    seed_strategy: SeedStrategy = Field(
        default=SeedStrategy.JOINT, validate_default=True
    )
    decimals: dict[str, int] | None = None

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

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

        general_input_dict = general_input.model_dump()
        general_input_dict.pop("rms_seeds")

        configuration = (
            general_input_dict
            | asdict(design_input)
            | {
                "background": background,
                "seeds": rms_seeds,
                "defaultvalues": default_values,
            }
        )

        return cls(**configuration)
