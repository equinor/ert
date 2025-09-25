from __future__ import annotations

from textwrap import dedent

from pydantic import BaseModel, Field, model_validator


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
