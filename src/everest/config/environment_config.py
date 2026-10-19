from textwrap import dedent
from typing import Literal, Self

from numpy.random import SeedSequence
from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator

from ert.config import ConfigWarning
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
            Folder for EVEREST output.
            """
        ),
    )
    log_level: Literal["debug", "info", "warning", "error", "critical"] | None = Field(
        default=None,
        deprecated=True,
        description=dedent(
            """
            Deprecated and ignored. The log level is no longer configurable;
            logs are written to disk at the `debug` level.
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
    def warn_about_deprecated_log_level(self) -> Self:
        if self.log_level is not None:
            ConfigWarning.deprecation_warn(
                "`environment: log_level` is deprecated and ignored. "
                "The log level is no longer configurable."
            )
        return self

    @model_validator(mode="after")
    def validate_random_seed(self) -> Self:
        if self.random_seed is None:
            self.random_seed = SeedSequence().entropy
        return self
