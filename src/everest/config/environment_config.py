from typing import Literal

from pydantic import BaseModel, Field, field_validator

from everest.config.validation_utils import check_path_valid


class EnvironmentConfig(BaseModel, extra="forbid"):  # type: ignore
    simulation_folder: str | None = Field(
        default="simulation_folder", description="Folder used for simulation by Everest"
    )
    output_folder: str | None = Field(
        default="everest_output", description="Folder for outputs of Everest"
    )
    log_level: Literal["debug", "info", "warning", "error", "critical"] | None = Field(
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
    random_seed: int | None = Field(
        default=None, description="Random seed (must be positive)"
    )

    @field_validator("output_folder", mode="before")
    @classmethod
    def validate_output_folder(cls, output_folder):  # pylint:disable=E0213
        check_path_valid(output_folder)
        return output_folder
