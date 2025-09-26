from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class InstallDataConfig(BaseModel):
    source: str | None = Field(
        default=None,
        description=dedent(
            """
            Path to file or directory that needs to be copied or linked into the
            directory where the forward model executes.
            """
        ),
    )
    target: str = Field(
        description=dedent(
            """
            Relative destination path to copy or link the given source to.
            """
        )
    )
    link: bool = Field(
        default=False,
        description=dedent(
            """
            If set to true will create a link to the given source at the given target,
            if not set the source will be copied at the given target.
            """
        ),
    )
    data: dict[str, Any] | None = Field(
        default=None,
        description=dedent(
            """
            Data to be exported to the target file.

            If provided, the data is exported to `target`, according to its file
            extension:

            - `.json`: Export to json.
            - `.yaml` or `.yml`: Export to yaml.

            If `target` has no extension, json format is used, and a `.json`
            extension is added.

            The `data` and `source` fields are mutually exclusive.

            If `data` is provided, `link` will be ignored.
            """
        ),
    )

    model_config = ConfigDict(
        extra="forbid",
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
    def validate_target(self) -> InstallDataConfig:
        if self.data is not None:
            if self.source is not None:
                raise ValueError("The data and source options are mutually exclusive.")
            if self.target.strip() in {"", ".", "./"}:
                raise ValueError("A target name must be provided with data.")

            ext = Path(self.target).suffix.lower()
            if ext not in {".json", ".yaml", ".yml", ""}:
                raise ValueError(f"Invalid target extension {ext}.")

            return self

        if self.source is None:
            raise ValueError("Either source or data must be provided.")

        if self.target.strip() in {".", "./"}:
            self.target = Path(self.source).name
        return self

    def inline_data_as_str(self) -> tuple[str, str]:
        if self.data is None:
            return "", ""
        match Path(self.target).suffix.lower():
            case ".json" | "":
                target = str(Path(self.target).with_suffix(".json"))
                data = json.dumps(self.data, indent=2)
            case ".yaml" | ".yml":
                target = self.target
                data = yaml.dump(self.data, sort_keys=False, default_flow_style=False)
            case _ as ext:
                raise ValueError(f"Invalid target extension {ext}.")
        return target, data
