from __future__ import annotations

import builtins
import logging
import os
import textwrap
from abc import ABC, abstractmethod
from dataclasses import field
from typing import TYPE_CHECKING, Self, TypeAlias, cast

from pydantic import field_serializer, field_validator, model_validator
from pydantic_core.core_schema import ValidationInfo

from ert.base_model_context import BaseModelWithContextSupport

from .ert_plugin import ErtPlugin
from .ert_script import ErtScript
from .parse_arg_types_list import parse_arg_types_list
from .parsing import (
    ConfigDict,
    ConfigWarning,
    SchemaItemType,
    WorkflowJobKeys,
    init_workflow_job_schema,
    parse,
)

if TYPE_CHECKING:
    from ert.plugins import ErtRuntimePlugins

logger = logging.getLogger(__name__)

ContentTypes: TypeAlias = type[int] | type[bool] | type[float] | type[str]


class ErtScriptLoadFailure(ValueError):
    pass


def workflow_job_parser(file: str) -> ConfigDict:
    schema = init_workflow_job_schema()
    return parse(file, schema=schema)


def workflow_job_from_file(config_file: str, name: str | None = None) -> WorkflowJob:
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


class _WorkflowJob(BaseModelWithContextSupport, ABC):
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
        )

        specified_max_args: int = content_dict.get("MAX_ARG", 0)
        specified_min_args: int = content_dict.get("MIN_ARG", 0)

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

    @abstractmethod
    def location(self) -> str | None: ...


class ExecutableWorkflow(_WorkflowJob):
    executable: str | None = None

    def location(self) -> str | None:
        return self.executable


class ErtScriptWorkflow(_WorkflowJob):
    """
    Single workflow configuration object
    """

    ert_script: builtins.type[ErtScript] = None  # type: ignore
    description: str = ""
    examples: str | None = None
    category: str = "other"

    @field_serializer("ert_script")
    def serialize_ert_script(self, _: str | builtins.type[ErtScript]) -> str:
        return self.name

    @field_validator("ert_script", mode="before")
    @classmethod
    def deserialize_ert_script(
        cls, ert_script: str | builtins.type[ErtScript], info: ValidationInfo
    ) -> builtins.type[ErtScript]:
        if isinstance(ert_script, type) and issubclass(ert_script, ErtScript):
            return ert_script

        runtime_plugins = cast("ErtRuntimePlugins", info.context)
        ertscript_workflow_job = runtime_plugins.installed_workflow_jobs.get(ert_script)

        if ertscript_workflow_job is None:
            raise KeyError(
                f"Did not find installed workflow job: {ert_script}. "
                f"installed workflow jobs are: "
                f"{runtime_plugins.installed_workflow_jobs.keys()}"
            )
        assert isinstance(ertscript_workflow_job, ErtScriptWorkflow)

        return ertscript_workflow_job.ert_script

    def location(self) -> str | None:
        return str(self.ert_script) if self.ert_script else None

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


WorkflowJob: TypeAlias = ErtScriptWorkflow | ExecutableWorkflow
