from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TypeAlias

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


def workflow_job_parser(file: str) -> ConfigDict:
    schema = init_workflow_job_schema()
    return parse(file, schema=schema)


class ErtScriptLoadFailure(ValueError):
    pass


@dataclass
class WorkflowJob:
    name: str
    min_args: int | None
    max_args: int | None
    arg_types: list[SchemaItemType]
    executable: str | None
    ert_script: type[ErtScript] | None = None
    stop_on_fail: bool | None = None  # If not None, overrides in-file specification

    def __post_init__(self) -> None:
        match self.ert_script:
            case None:
                pass
            case ert_script if not isinstance(ert_script, type):
                raise ErtScriptLoadFailure(
                    f"Failed to load {self.name}, ert_script is instance, expected "
                    f"type, got {ert_script}"
                )
            case ert_script if not issubclass(ert_script, ErtScript):
                raise ErtScriptLoadFailure(
                    f"Failed to load {self.name}, script had wrong "
                    f"type, expected ErtScript, got {ert_script}"
                )

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

    @classmethod
    def from_file(cls, config_file: str, name: str | None = None) -> WorkflowJob:
        if not name:
            name = os.path.basename(config_file)

        content_dict = workflow_job_parser(config_file)
        arg_types_list = cls._make_arg_types_list(content_dict)
        script = str(content_dict.get("SCRIPT")) if "SCRIPT" in content_dict else None  # type: ignore
        internal = (
            bool(content_dict.get("INTERNAL")) if "INTERNAL" in content_dict else None  # type: ignore
        )
        ert_script = None
        if internal is False:
            ConfigWarning.deprecation_warn(
                "INTERNAL FALSE has no effect and can be safely removed",
                content_dict["INTERNAL"],  # type: ignore
            )
        if script and not internal:
            ConfigWarning.deprecation_warn(
                "SCRIPT has no effect and can be safely removed",
                content_dict["SCRIPT"],  # type: ignore
            )
        elif script is not None and internal:
            msg = f"Deprecated keywords, SCRIPT and INTERNAL, for {name}, loading script {script}"
            logger.warning(msg)
            ConfigWarning.deprecation_warn(msg, content_dict["SCRIPT"])  # type: ignore
            try:
                ert_script = ErtScript.loadScriptFromFile(script)
            # Bare Exception here as we have no control
            # of exceptions in the loaded ErtScript
            except Exception as err:
                raise ErtScriptLoadFailure(f"Failed to load {name}: {err}") from err

        return cls(
            name=name,
            min_args=content_dict.get("MIN_ARG"),  # type: ignore
            max_args=content_dict.get("MAX_ARG"),  # type: ignore
            arg_types=arg_types_list,
            executable=content_dict.get("EXECUTABLE"),  # type: ignore
            stop_on_fail=content_dict.get("STOP_ON_FAIL"),  # type: ignore
            ert_script=ert_script,  # type: ignore
        )

    def is_plugin(self) -> bool:
        if self.ert_script is not None:
            return issubclass(self.ert_script, ErtPlugin)
        return False

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
