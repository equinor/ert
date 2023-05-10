from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Type, Union, Mapping

from ert._c_wrappers.config import ConfigParser, ContentTypeEnum
from ert._clib.job_kw import type_from_kw
from ert.parsing import ConfigValidationError, lark_parse

from .ert_plugin import ErtPlugin
from .ert_script import ErtScript
from ...parsing.schema_item_type import SchemaItemType
from ...parsing.types import Instruction
from ...parsing.workflow_job_keywords import WorkflowJobKeys
from ...parsing.workflow_job_schema import init_workflow_schema

ContentTypes = Union[Type[int], Type[bool], Type[float], Type[str]]

USE_NEW_PARSER_BY_DEFAULT = True

if "USE_NEW_ERT_PARSER" in os.environ and os.environ["USE_NEW_ERT_PARSER"] == "YES":
    USE_NEW_PARSER_BY_DEFAULT = True


def _workflow_job_config_parser() -> ConfigParser:
    parser = ConfigParser()
    parser.add("MIN_ARG", value_type=ContentTypeEnum.CONFIG_INT).set_argc_minmax(1, 1)
    parser.add("MAX_ARG", value_type=ContentTypeEnum.CONFIG_INT).set_argc_minmax(1, 1)
    parser.add(
        "EXECUTABLE", value_type=ContentTypeEnum.CONFIG_EXECUTABLE
    ).set_argc_minmax(1, 1)
    parser.add("SCRIPT", value_type=ContentTypeEnum.CONFIG_PATH).set_argc_minmax(1, 1)
    parser.add("INTERNAL", value_type=ContentTypeEnum.CONFIG_BOOL).set_argc_minmax(1, 1)
    item = parser.add("ARG_TYPE")
    item.set_argc_minmax(2, 2)
    item.iset_type(0, ContentTypeEnum.CONFIG_INT)
    item.initSelection(1, ["STRING", "INT", "FLOAT", "BOOL"])
    return parser


_config_parser = _workflow_job_config_parser()


def supreme_parser(file: str):
    best_schema = init_workflow_schema()
    return lark_parse(file, schema=best_schema)


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
    def _make_arg_types_list_new(content_dict: Mapping[str, Instruction]):
        arg_types_dict = defaultdict(lambda _: "STRING")
        args_spec = content_dict.get(WorkflowJobKeys.ARG_TYPE, [])

        if len(args_spec) == 0:
            return [], 0

        for i, type_as_string in args_spec:
            # make old tests pass temporarily, will use
            # SchemaItemType
            arg_types_dict[i] = WorkflowJob.stringToType(type_as_string)

        types_list = [arg_types_dict[i] for i in range(max(arg_types_dict.keys()) + 1)]
        max_argc_from_type_list = len(types_list)
        max_argc_from_dict = content_dict.get(
            WorkflowJobKeys.MAX_ARG, max_argc_from_type_list
        )

        if max_argc_from_dict < max_argc_from_type_list:
            raise ConfigValidationError(
                f"Argument number {max_argc_from_type_list - 1} exceeds specified "
                f"max number of arguments ({max_argc_from_dict})"
            )

        return types_list, max_argc_from_dict

    @staticmethod
    def _make_arg_types_list(
        config_content, max_arg: Optional[int]
    ) -> List[ContentTypeEnum]:
        arg_types_dict = defaultdict(lambda: ContentTypeEnum.CONFIG_STRING)
        if max_arg is not None:
            arg_types_dict[max_arg - 1] = ContentTypeEnum.CONFIG_STRING
        for arg in config_content["ARG_TYPE"]:
            arg_types_dict[arg[0]] = ContentTypeEnum(type_from_kw(arg[1]))
        if arg_types_dict:
            return [
                arg_types_dict[j]
                for j in range(max(i for i in arg_types_dict.keys()) + 1)
            ]
        else:
            return []

    @classmethod
    def fromFile(cls, config_file, name=None):
        if os.path.isfile(config_file) and os.access(config_file, os.R_OK):
            if name is None:
                name = os.path.basename(config_file)

            old_content = _config_parser.parse(config_file)

            def optional_get(key):
                return old_content.getValue(key) if old_content.hasKey(key) else None

            max_arg = optional_get("MAX_ARG")

            old_arg_types_list = cls._make_arg_types_list(old_content, max_arg)
            old_job = cls(
                name,
                optional_get("INTERNAL"),
                optional_get("MIN_ARG"),
                max_arg,
                old_arg_types_list,
                optional_get("EXECUTABLE"),
                optional_get("SCRIPT"),
            )

            new_content_dict = supreme_parser(config_file)

            new_types_list, new_max_argc = cls._make_arg_types_list_new(
                new_content_dict
            )
            new_job = cls(
                name,
                new_content_dict.get("INTERNAL", None),
                new_content_dict.get("MIN_ARG", None),
                new_max_argc,
                new_types_list,
                new_content_dict.get("EXECUTABLE", None),
                new_content_dict.get("SCRIPT", None),
            )

            return new_job
        else:
            raise ConfigValidationError(f"Could not open config_file:{config_file!r}")

    def isPlugin(self) -> bool:
        if self.ert_script is not None:
            return issubclass(self.ert_script, ErtPlugin)
        return False

    @classmethod
    def stringToType(cls, string: str):
        if string == "STRING":
            return ContentTypeEnum.CONFIG_STRING
        if string == "FLOAT":
            return ContentTypeEnum.CONFIG_FLOAT
        if string == "INT":
            return ContentTypeEnum.CONFIG_INT
        if string == "BOOL":
            return ContentTypeEnum.CONFIG_BOOL

        raise ValueError(f"Unknown content type")

    def argumentTypes(self) -> List["ContentTypes"]:
        def content_to_type(c: Optional[ContentTypeEnum]):
            if c == ContentTypeEnum.CONFIG_BOOL:
                return bool
            if c == ContentTypeEnum.CONFIG_FLOAT:
                return float
            if c == ContentTypeEnum.CONFIG_INT:
                return int
            if c == ContentTypeEnum.CONFIG_STRING:
                return str
            raise ValueError(f"Unknown job type {c} in {self}")

        return list(map(content_to_type, self.arg_types))
