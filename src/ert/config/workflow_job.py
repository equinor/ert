from __future__ import annotations

import logging
import os
import textwrap
from argparse import ArgumentParser
from collections.abc import Callable
from dataclasses import field
from typing import Self, TypeAlias

from pydantic import BaseModel, model_validator

from ..plugins.ert_plugin import ErtPlugin
from ..plugins.ert_script import ErtScript
from .parse_arg_types_list import parse_arg_types_list
from .parsing import (
    ConfigDict,
    SchemaItemType,
    WorkflowJobKeys,
    init_workflow_job_schema,
    parse,
)
from .parsing.config_errors import ConfigWarning

logger = logging.getLogger(__name__)

ContentTypes: TypeAlias = type[int] | type[bool] | type[float] | type[str]


class ErtScriptLoadFailure(ValueError):
    pass


def workflow_job_parser(file: str) -> ConfigDict:
    schema = init_workflow_job_schema()
    return parse(file, schema=schema)


def workflow_job_from_file(config_file: str, name: str | None = None) -> _WorkflowJob:
    if not name:
        name = os.path.basename(config_file)

    content_dict = workflow_job_parser(config_file)
    arg_types_list = _WorkflowJob._make_arg_types_list(content_dict)
    min_args = content_dict.get("MIN_ARG")
    max_args = content_dict.get("MAX_ARG")
    script = str(content_dict.get("SCRIPT")) if "SCRIPT" in content_dict else None
    internal = (
        bool(content_dict.get("INTERNAL")) if "INTERNAL" in content_dict else None
    )
    if internal is False:
        ConfigWarning.deprecation_warn(
            "INTERNAL FALSE has no effect and can be safely removed",
            content_dict["INTERNAL"],
        )
    if script and not internal:
        ConfigWarning.deprecation_warn(
            "SCRIPT has no effect and can be safely removed",
            content_dict["SCRIPT"],
        )
    if script is not None and internal:
        msg = (
            "Deprecated keywords, SCRIPT and INTERNAL, "
            f"for {name}, loading script {script}"
        )
        logger.warning(msg)
        ConfigWarning.deprecation_warn(msg, content_dict["SCRIPT"])
        try:
            ert_script = ErtScript.loadScriptFromFile(script)
        # Bare Exception here as we have no control
        # of exceptions in the loaded ErtScript
        except Exception as err:
            raise ErtScriptLoadFailure(f"Failed to load {name}: {err}") from err
        return ErtScriptWorkflow(
            ert_script=ert_script,
            name=name,
            min_args=min_args,
            max_args=max_args,
            arg_types=arg_types_list,
            stop_on_fail=bool(content_dict.get("STOP_ON_FAIL")),
        )
    else:
        return ExecutableWorkflow(
            name=name,
            min_args=min_args,
            max_args=max_args,
            arg_types=arg_types_list,
            executable=content_dict.get("EXECUTABLE"),
            stop_on_fail=bool(content_dict.get("STOP_ON_FAIL")),
        )


class _WorkflowJob(BaseModel):
    name: str
    min_args: int | None = None
    max_args: int | None = None
    arg_types: list[SchemaItemType] = field(default_factory=list)
    stop_on_fail: bool = False

    @staticmethod
    def _make_arg_types_list(content_dict: ConfigDict) -> list[SchemaItemType]:
        # First find the number of args
        specified_arg_types: list[tuple[int, str]] = content_dict.get(
            WorkflowJobKeys.ARG_TYPE, []
        )  # type: ignore

        specified_max_args: int = content_dict.get("MAX_ARG", 0)  # type: ignore
        specified_min_args: int = content_dict.get("MIN_ARG", 0)  # type: ignore

        return parse_arg_types_list(
            specified_arg_types, specified_min_args, specified_max_args
        )

    def argument_types(self) -> list[ContentTypes]:
        def content_to_type(c: SchemaItemType | None) -> ContentTypes:
            if c == SchemaItemType.BOOL:
                return bool
            if c == SchemaItemType.FLOAT:
                return float
            if c == SchemaItemType.INT:
                return int
            if c == SchemaItemType.STRING:
                return str
            raise ValueError(f"Unknown job type {c} in {self}")

        return list(map(content_to_type, self.arg_types))

    def is_plugin(self) -> bool:
        return False


class ExecutableWorkflow(_WorkflowJob):
    executable: str | None = None


class ErtScriptWorkflow(_WorkflowJob):
    """
    Single workflow configuration object
    """

    ert_script: type[ErtScript] = None  # type: ignore
    description: str = ""
    examples: str | None = None
    parser: Callable[[], ArgumentParser] | None = None
    category: str = "other"

    @model_validator(mode="after")
    def validate_types(self) -> Self:
        if not isinstance(self.ert_script, type):
            raise ErtScriptLoadFailure(
                f"Failed to load {self.name}, ert_script is instance, expected "
                f"type, got {self.ert_script}"
            )
        elif not issubclass(self.ert_script, ErtScript):
            raise ErtScriptLoadFailure(
                f"Failed to load {self.name}, script had wrong "
                f"type, expected ErtScript, got {self.ert_script}"
            )
        if self.ert_script.__doc__ is not None:
            self.description = textwrap.dedent(self.ert_script.__doc__.strip())
        return self

    @property
    def source_package(self) -> str:
        return self.ert_script.__module__.partition(".")[2]

    def is_plugin(self) -> bool:
        return issubclass(self.ert_script, ErtPlugin)
