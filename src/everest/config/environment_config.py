from typing import Literal, Self

from numpy.random import SeedSequence
from pydantic import BaseModel, Field, field_validator, model_validator

from everest.config.validation_utils import check_path_valid


class EnvironmentConfig(BaseModel, extra="forbid"):
    simulation_folder: str = Field(
        default="simulation_folder", description="Folder used for simulation by Everest"
    )
    output_folder: str = Field(
        default="everest_output", description="Folder for outputs of Everest"
    )
    log_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info",
        description="""Defines the verbosity of logs output by Everest.

The default log level is `info`. All supported log levels are:

debug: Detailed information, typically of interest only when diagnosing
problems.

info: Confirmation that things are working as expected.

warning: An indication that something unexpected happened, or indicative of some
problem in the near future (e.g. `disk space low`). The software is still
working as expected.

error: Due to a more serious problem, the software has not been able to perform
some function.

critical: A serious error, indicating that the program itself may be unable to
continue running.
""",
    )
    random_seed: int = Field(default=None, description="Random seed (must be positive)")  # type: ignore

    @field_validator("output_folder", mode="before")
    @classmethod
    def validate_output_folder(cls, output_folder: str | None) -> str:
        if output_folder is None:
            raise ValueError("output_folder can not be None")
        check_path_valid(output_folder)
        return output_folder

    @model_validator(mode="after")
    def validate_random_seed(self) -> Self:
        if self.random_seed is None:
            self.random_seed = SeedSequence().entropy
        return self
