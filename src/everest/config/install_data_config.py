from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from textwrap import dedent
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ert.config.forward_model_step import (
    SiteOrUserForwardModelStep,
)


def _is_dir_all_model(source: str, model_realizations: list[int]) -> bool:
    """Expands <GEO_ID> for all realizations and if:
    - all are directories, returns True,
    - all are files, returns False,
    - some are non-existing, raises an AssertionError
    """

    is_dir = []
    for model_realization in model_realizations:
        model_source = source.replace("<GEO_ID>", str(model_realization))
        if not os.path.exists(model_source):
            msg = (
                "Expected source to exist for data installation, "
                f"did not find: {model_source}"
            )
            raise ValueError(msg)

        is_dir.append(os.path.isdir(model_source))

    if set(is_dir) == {True, False}:
        msg = f"Source: {source} represent both files and directories"
        raise ValueError(msg)

    return is_dir[0]


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

            If provided, the data is exported to a JSON file. If `target` has no
            `.json` extension, it is added.

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
            if ext not in {".json", ""}:
                raise ValueError(f"Invalid target extension {ext} (.json expected).")

            return self

        if self.source is None:
            raise ValueError("Either source or data must be provided.")

        if self.target.strip() in {".", "./"}:
            self.target = Path(self.source).name
        return self

    def inline_data_as_str(self) -> tuple[str, str]:
        assert self.data is not None
        return (
            str(Path(self.target).with_suffix(".json")),
            json.dumps(self.data, indent=2),
        )

    def to_ert_forward_model_step(
        self,
        config_directory: str,
        output_directory: str,
        model_realizations: list[int],
        installed_fm_steps: dict[str, SiteOrUserForwardModelStep],
    ) -> SiteOrUserForwardModelStep:
        target = self.target

        def _missing_fm_msg(fm_name: str) -> str:
            return (
                f"Using install_data in Everest requires the "
                f"ERT forward model: {fm_name} to be installed. "
                f"It was not found in the installed forward model steps: "
                + (", ".join(installed_fm_steps.keys()))
            )

        if self.source is not None:
            source = self.source.replace("<CONFIG_PATH>", config_directory)
            if not os.path.isabs(source):
                source = os.path.join(config_directory, source)

            is_dir = _is_dir_all_model(source, model_realizations)

            if self.link:
                fm_name = "symlink"
            elif is_dir:
                fm_name = "copy_directory"
            else:
                fm_name = "copy_file"

            fm_step_instance = copy.deepcopy(installed_fm_steps.get(fm_name))
            if fm_step_instance is None:
                error_message = _missing_fm_msg(fm_name)
                raise KeyError(error_message)

            fm_step_instance.arglist = [source, target]
            return fm_step_instance
        else:
            data_storage = (Path(output_directory) / ".internal_data").resolve()
            data_file = data_storage / Path(self.target).with_suffix(".json")
            fm_step_instance = copy.deepcopy(installed_fm_steps.get("copy_file"))
            if fm_step_instance is None:
                error_message = _missing_fm_msg("copy_file")
                raise KeyError(error_message)

            fm_step_instance.arglist = [str(data_file), Path(data_file).name]
            return fm_step_instance
