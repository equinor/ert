from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, Union

from ert._c_wrappers.config import ConfigParser
from ert._clib.job_kw import type_from_kw
from ert.parsing import (
    ConfigDict,
    ConfigValidationError,
    SchemaItemType,
    WorkflowJobKeys,
    init_workflow_schema,
    lark_parse,
)

from .ert_plugin import ErtPlugin
from .ert_script import ErtScript

logger = logging.getLogger(__name__)

ContentTypes = Union[Type[int], Type[bool], Type[float], Type[str]]


def _workflow_job_config_parser() -> ConfigParser:
    parser = ConfigParser()
    parser.add("MIN_ARG", value_type=SchemaItemType.INT).set_argc_minmax(1, 1)
    parser.add("MAX_ARG", value_type=SchemaItemType.INT).set_argc_minmax(1, 1)
    parser.add("EXECUTABLE", value_type=SchemaItemType.EXECUTABLE).set_argc_minmax(1, 1)
    parser.add("SCRIPT", value_type=SchemaItemType.PATH).set_argc_minmax(1, 1)
    parser.add("INTERNAL", value_type=SchemaItemType.BOOL).set_argc_minmax(1, 1)
    item = parser.add("ARG_TYPE")
    item.set_argc_minmax(2, 2)
    item.iset_type(0, SchemaItemType.INT)
    item.initSelection(1, ["STRING", "INT", "FLOAT", "BOOL"])
    return parser


_config_parser = _workflow_job_config_parser()


def new_workflow_job_parser(file: str):
    schema = init_workflow_schema()
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

    def __post_init__(self):
        self.ert_script: Optional[type] = None
        if self.script is not None and self.internal:
            try:
                self.ert_script = ErtScript.loadScriptFromFile(
                    self.script,
                )  # type: ignore
            # Bare Exception here as we have no control
            # of exceptions in the loaded ErtScript
            except Exception as err:  # noqa
                raise ErtScriptLoadFailure(
                    f"Failed to load {self.name}: {err}"
                ) from err

    @staticmethod
    def _make_arg_types_list_new(content_dict: ConfigDict) -> List[SchemaItemType]:
        # First find the number of args
        args_spec: List[Tuple[int, str]] = content_dict.get(
            WorkflowJobKeys.ARG_TYPE, []
        )  # type: ignore
        specified_highest_arg_index = (
            max(index for index, _ in args_spec) if len(args_spec) > 0 else -1
        )
        specified_max_args: int = content_dict.get("MAX_ARG", 0)  # type: ignore
        specified_min_args: int = content_dict.get("MIN_ARG", 0)  # type: ignore

        num_args = max(
            specified_highest_arg_index + 1,
            specified_max_args,
            specified_min_args,
        )

        arg_types_dict: Dict[int, SchemaItemType] = dict()

        for i, type_as_string in args_spec:
            arg_types_dict[i] = WorkflowJob.string_to_type(type_as_string)

        arg_types_list: List[SchemaItemType] = [
            arg_types_dict.get(i, SchemaItemType.STRING) for i in range(num_args)
        ]  # type: ignore
        return arg_types_list

    @staticmethod
    def _make_arg_types_list(
        config_content, max_arg: Optional[int]
    ) -> List[SchemaItemType]:
        arg_types_dict: Dict[int, SchemaItemType] = defaultdict(
            lambda: SchemaItemType.STRING
        )
        if max_arg is not None:
            arg_types_dict[max_arg - 1] = SchemaItemType.STRING
        for arg in config_content["ARG_TYPE"]:
            arg_types_dict[arg[0]] = SchemaItemType.from_content_type_enum(
                type_from_kw(arg[1])
            )
        if arg_types_dict:
            return [
                arg_types_dict[j]
                for j in range(max(i for i in arg_types_dict.keys()) + 1)
            ]
        else:
            return []

    @classmethod
    def from_file(cls, config_file, name=None, use_new_parser=True):
        if not (os.path.isfile(config_file) and os.access(config_file, os.R_OK)):
            raise ConfigValidationError(f"Could not open config_file:{config_file!r}")
        if not name:
            name = os.path.basename(config_file)

        if use_new_parser:
            content_dict = new_workflow_job_parser(config_file)
            arg_types_list = cls._make_arg_types_list_new(content_dict)
            return cls(
                name,
                content_dict.get("INTERNAL"),  # type: ignore
                content_dict.get("MIN_ARG"),  # type: ignore
                content_dict.get("MAX_ARG"),  # type: ignore
                arg_types_list,
                content_dict.get("EXECUTABLE"),  # type: ignore
                str(content_dict.get("SCRIPT")) if "SCRIPT" in content_dict else None,
            )
        else:
            old_content = _config_parser.parse(config_file)

            def optional_get(key, default=None):
                return old_content.getValue(key) if old_content.hasKey(key) else default

            max_arg = optional_get("MAX_ARG")
            arg_types_list = cls._make_arg_types_list(old_content, max_arg)
            return cls(
                name,
                optional_get("INTERNAL", False),
                optional_get("MIN_ARG"),
                max_arg,
                arg_types_list,
                optional_get("EXECUTABLE"),
                optional_get("SCRIPT"),
            )

    def is_plugin(self) -> bool:
        if self.ert_script is not None:
            return issubclass(self.ert_script, ErtPlugin)
        return False

    @classmethod
    def string_to_type(cls, string: str) -> SchemaItemType:
        if string == "STRING":
            return SchemaItemType.STRING
        if string == "FLOAT":
            return SchemaItemType.FLOAT
        if string == "INT":
            return SchemaItemType.INT
        if string == "BOOL":
            return SchemaItemType.BOOL

        raise ValueError("Unrecognized content type")

    def argument_types(self) -> List["ContentTypes"]:
        def content_to_type(c: Optional[SchemaItemType]):
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
