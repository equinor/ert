import copy
import os
from pathlib import Path
from textwrap import dedent

from pydantic import BaseModel, Field, field_validator, model_validator

from ert.config import ForwardModelStep


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

    def to_ert_forward_model_step(
        self,
        config_directory: str,
        model_realizations: list[int],
        installed_fm_steps: dict[str, ForwardModelStep],
    ) -> ForwardModelStep:
        target = self.target

        source = self.source.replace("<CONFIG_PATH>", config_directory)
        if not os.path.isabs(source):
            source = os.path.join(config_directory, source)

        is_dir = _is_dir_all_model(source, model_realizations)

        fm_name: str | None = None
        if self.link:
            fm_name = "symlink"
        elif is_dir:
            fm_name = "copy_directory"
        else:
            fm_name = "copy_file"

        assert isinstance(fm_name, str)

        fm_step_instance = copy.deepcopy(installed_fm_steps.get(fm_name))
        assert fm_step_instance is not None
        fm_step_instance.arglist = [source, target]
        return fm_step_instance
