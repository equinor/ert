from pathlib import Path
from textwrap import dedent

from pydantic import BaseModel, Field, field_validator, model_validator


class InstallDataConfig(BaseModel, extra="forbid"):
    source: str = Field(
        description=dedent(
            """
            Path to file or directory that needs to be copied or linked into the
            directory where the forward model executes.
            """
        )
    )  # existing path
    target: str = Field(
        description=dedent(
            """
            Relative destination path to copy or link the given source to.
            """
        )
    )  # path
    link: bool = Field(
        default=False,
        description=dedent(
            """
            If set to true will create a link to the given source at the given target,
            if not set the source will be copied at the given target.
            """
        ),
    )

    @field_validator("link", mode="before")
    @classmethod
    def validate_link_type(cls, link: bool | None) -> bool | None:
        if link is None:
            return None
        if not isinstance(link, bool):
            raise ValueError(f" {link} could not be parsed to a boolean")
        return link

    @model_validator(mode="after")
    def validate_target(self) -> "InstallDataConfig":
        if self.target in {".", "./"}:
            self.target = Path(self.source).name
        return self
