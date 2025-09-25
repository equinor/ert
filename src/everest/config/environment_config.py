from textwrap import dedent
from typing import Literal, Self

from numpy.random import SeedSequence
from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator

from everest.config.validation_utils import check_path_valid


class EnvironmentConfig(BaseModel, extra="forbid"):
    simulation_folder: str = Field(
        default="simulation_folder",
        description=dedent(
            """
            Folder where the forward model outputs are stored.

            It the provided path is not absolute, it is assumed to be relative
            with respect to the path given by `output_folder`.
            """
        ),
    )
    output_folder: str = Field(
        default="everest_output",
        description=dedent(
            """
            Folder for Everest output.
            """
        ),
    )
    log_level: Literal["debug", "info", "warning", "error", "critical"] = Field(
        default="info",
        description=dedent(
            """
            Defines the verbosity of logs output by Everest.

            The default log level is `info`. The supported log levels are:

            - `debug`: Detailed information, typically of interest only when
               diagnosing problems.
            - `info`: Confirmation that things are working as expected.
            - `warning`: An indication that something unexpected happened, or
               indicative of some problem in the near future (e.g. `disk space
               low`). The software is still working as expected.
            - `error`: Due to a more serious problem, the software has not been
               able to perform some function.
            - `critical`: A serious error, indicating that the program itself
               may be unable to continue running.
            """
        ),
    )
    random_seed: PositiveInt = Field(
        default=None,
        description=dedent(
            """
            Random seed.

            A positive integer used to seed the random number generator that is
            used to generate perturbed controls for gradient estimation.
            """
        ),
    )  # type: ignore

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
