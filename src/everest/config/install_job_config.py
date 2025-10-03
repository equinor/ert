from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from pydantic import BaseModel, Field, model_validator

from ert.config import ForwardModelStep, forward_model_step_from_config_contents


class InstallJobConfig(BaseModel, extra="forbid"):
    name: str = Field(
        description=dedent(
            """
            The name of the installed job.

            This name is used to identify the job in the list of jobs to be
            executed, as defined by the `forward_model` field.
            """
        )
    )
    source: str | None = Field(
        default=None,
        description=dedent(
            """
            The source file of the ert job.

            `source` is deprecated, please use `executable` instead.
            """
        ),
    )
    executable: str | None = Field(
        default=None,
        description=dedent(
            """
            The executable linked to the job name.
            """
        ),
    )

    @model_validator(mode="after")
    def validate_source_and_executable(self) -> InstallJobConfig:
        if self.source is None and self.executable is None:
            raise ValueError("Either source or executable must be provided")
        return self

    def to_ert_forward_model_step(self, config_directory: str) -> ForwardModelStep:
        if self.executable is not None:
            executable = Path(self.executable)
            if not executable.is_absolute():
                executable = Path(config_directory) / executable
            return ForwardModelStep(name=self.name, executable=str(executable))
        else:
            assert (
                self.source is not None
            )  # validated by validate_source_and_executable
            return forward_model_step_from_config_contents(
                config_contents=Path(self.source).read_text(encoding="utf-8"),
                config_file=self.source,
                name=self.name,
            )
