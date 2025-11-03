from __future__ import annotations

import builtins
import logging
import os
import textwrap
from abc import ABC, abstractmethod
from dataclasses import field
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Self,
    TypeAlias,
    cast,
)

from pydantic import (
    Field,
    model_serializer,
    model_validator,
)
from pydantic_core.core_schema import ValidationInfo
from typing_extensions import TypedDict

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


def workflow_job_from_file(
    config_file: str,
    origin: Literal["user", "site"] = "site",
    name: str | None = None,
) -> WorkflowJob:
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

        if origin == "user":
            return UserInstalledErtScriptWorkflow(
                name=name,
                min_args=min_args,
                max_args=max_args,
                arg_types=arg_types_list,
                source=script,
                stop_on_fail=bool(content_dict.get("STOP_ON_FAIL")),
            )
        else:
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
    type: Literal["user_installed_executable"] = "user_installed_executable"
    executable: str | None = None

    def location(self) -> str | None:
        return self.executable


class BaseErtScriptWorkflow(_WorkflowJob, ABC):
    @abstractmethod
    def load_ert_script_class(self) -> builtins.type[ErtScript]: ...

    def location(self) -> str | None:
        return (
            str(self.load_ert_script_class()) if self.load_ert_script_class() else None
        )

    @model_validator(mode="after")
    def validate_types(self) -> Self:
        ertscript_class = self.load_ert_script_class()
        if not isinstance(ertscript_class, type):
            raise ErtScriptLoadFailure(
                f"Failed to load {self.name}, ert_script is instance, expected "
                f"type, got {ertscript_class}"
            )
        elif not issubclass(ertscript_class, ErtScript):
            raise ErtScriptLoadFailure(
                f"Failed to load {self.name}, script had wrong "
                f"type, expected ErtScript, got {ertscript_class}"
            )
        if ertscript_class.__doc__:
            self.description = textwrap.dedent(ertscript_class.__doc__.strip())
        return self

    @property
    def source_package(self) -> str:
        return self.load_ert_script_class().__module__.partition(".")[2]

    def is_plugin(self) -> bool:
        return issubclass(self.load_ert_script_class(), ErtPlugin)


class _SerializedSiteInstalledErtScriptWorkflow(TypedDict):
    type: Literal["site_installed"]
    name: str


class SiteInstalledErtScriptWorkflow(BaseErtScriptWorkflow):
    """
    Single workflow configuration object installed from site plugins
    """

    type: Literal["site_installed"] = "site_installed"
    ert_script: builtins.type[ErtScript] = None  # type: ignore
    description: str = ""
    examples: str | None = None
    category: str = "other"

    @model_serializer(mode="plain")
    def serialize_model(self) -> _SerializedSiteInstalledErtScriptWorkflow:
        return {"type": "site_installed", "name": self.name}

    def load_ert_script_class(self) -> builtins.type[ErtScript]:
        return self.ert_script

    @model_validator(mode="before")
    @classmethod
    def deserialize_model(
        cls, values: dict[str, Any], info: ValidationInfo
    ) -> dict[str, Any]:
        runtime_plugins = cast("ErtRuntimePlugins", info.context)
        name = values["name"]

        if runtime_plugins is None:
            if set(values.keys()) == {"name", "type"}:
                raise ValueError(
                    f"Cannot resolve workflow job {values},"
                    f"as it expects a the workflow job {name}"
                    f"to be installed."
                )
            return values

        if name not in runtime_plugins.installed_workflow_jobs:
            raise KeyError(
                f"Expected workflow job {name} to be installed "
                f"via plugins, but it was not found. Please check that "
                f"your python environment has it installed."
            )
        site_installed_wfjob = runtime_plugins.installed_workflow_jobs[name]

        # Intent: copy the site installed workflow to this instance.
        # bypassing the model_serializer
        return {
            k: getattr(site_installed_wfjob, k)
            for k in SiteInstalledErtScriptWorkflow.model_fields
        }


# We keep the old name for compatability with .legacy_ertscript_workflow
# all of which add ErtScriptWorkflow (always through plugins, i.e., site-installed)
ErtScriptWorkflow = SiteInstalledErtScriptWorkflow


class UserInstalledErtScriptWorkflow(BaseErtScriptWorkflow):
    type: Literal["user_installed_ertscript"] = "user_installed_ertscript"
    source: str

    def load_ert_script_class(self) -> builtins.type[ErtScript]:
        try:
            return ErtScript.loadScriptFromFile(self.source)
        except Exception as err:
            raise ErtScriptLoadFailure(f"Failed to load {self.name}: {err}") from err


WorkflowJob: TypeAlias = Annotated[
    (
        SiteInstalledErtScriptWorkflow
        | UserInstalledErtScriptWorkflow
        | ExecutableWorkflow
    ),
    Field(discriminator="type"),
]
