from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class InstallJobConfig(BaseModel, extra="forbid"):
    name: str = Field(description="name of the installed job")
    source: str | None = Field(
        default=None,
        description="""source file of the ert job.

`source` will be deprecated, please use `executable` instead.
    """,
    )
    executable: str | None = Field(default=None, description="Executable to run")

    @model_validator(mode="after")
    def validate_source_and_executable(self) -> InstallJobConfig:
        if self.source is None and self.executable is None:
            raise ValueError("Either source or executable must be provided")
        return self
