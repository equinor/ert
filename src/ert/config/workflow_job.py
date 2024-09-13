from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Type, Union

from .ert_plugin import ErtPlugin
from .ert_script import ErtScript
from .parse_arg_types_list import parse_arg_types_list
from .parsing import (
    ConfigDict,
    ConfigValidationError,
    SchemaItemType,
    WorkflowJobKeys,
    init_workflow_job_schema,
    lark_parse,
)

logger = logging.getLogger(__name__)

ContentTypes = Union[Type[int], Type[bool], Type[float], Type[str]]


def workflow_job_parser(file: str) -> ConfigDict:
    schema = init_workflow_job_schema()
    return lark_parse(file, schema=schema)


class ErtScriptLoadFailure(ValueError):
    pass


@dataclass
class WorkflowJob:
    name: str
    internal: bool
    min_args: Optional[int]
    max_args: Optional[int]
    arg_types: List[SchemaItemType]
    executable: Optional[str]
    script: Optional[str]
    stop_on_fail: Optional[bool] = None  # If not None, overrides in-file specification

    def __post_init__(self) -> None:
        self.ert_script: Optional[type] = None
        if self.script is not None and self.internal:
            try:
                self.ert_script = ErtScript.loadScriptFromFile(
                    self.script,
                )  # type: ignore

            # Bare Exception here as we have no control
            # of exceptions in the loaded ErtScript
            except Exception as err:
                raise ErtScriptLoadFailure(
                    f"Failed to load {self.name}: {err}"
                ) from err

    @staticmethod
    def _make_arg_types_list(content_dict: ConfigDict) -> List[SchemaItemType]:
        # First find the number of args
        specified_arg_types: List[Tuple[int, str]] = content_dict.get(
            WorkflowJobKeys.ARG_TYPE, []
        )  # type: ignore

        specified_max_args: int = content_dict.get("MAX_ARG", 0)  # type: ignore
        specified_min_args: int = content_dict.get("MIN_ARG", 0)  # type: ignore

        return parse_arg_types_list(
            specified_arg_types, specified_min_args, specified_max_args
        )

    @classmethod
    def from_file(cls, config_file: str, name: Optional[str] = None) -> "WorkflowJob":
        if not (os.path.isfile(config_file) and os.access(config_file, os.R_OK)):
            raise ConfigValidationError(f"Could not open config_file:{config_file!r}")
        if not name:
            name = os.path.basename(config_file)

        content_dict = workflow_job_parser(config_file)
        arg_types_list = cls._make_arg_types_list(content_dict)
        return cls(
            name=name,
            internal=content_dict.get("INTERNAL"),  # type: ignore
            min_args=content_dict.get("MIN_ARG"),  # type: ignore
            max_args=content_dict.get("MAX_ARG"),  # type: ignore
            arg_types=arg_types_list,
            executable=content_dict.get("EXECUTABLE"),  # type: ignore
            script=(
                str(content_dict.get("SCRIPT"))  # type: ignore
                if "SCRIPT" in content_dict
                else None
            ),
            stop_on_fail=content_dict.get("STOP_ON_FAIL"),  # type: ignore
        )

    def is_plugin(self) -> bool:
        if self.ert_script is not None:
            return issubclass(self.ert_script, ErtPlugin)
        return False

    def argument_types(self) -> List["ContentTypes"]:
        def content_to_type(c: Optional[SchemaItemType]) -> ContentTypes:
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
