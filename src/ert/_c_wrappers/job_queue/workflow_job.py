from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Mapping, Optional, Type, Union

from ert._c_wrappers.config import ConfigParser, ContentTypeEnum
from ert._clib.job_kw import type_from_kw
from ert.parsing import (
    ConfigValidationError,
    Instruction,
    WorkflowJobKeys,
    init_workflow_schema,
    lark_parse,
)

from .ert_plugin import ErtPlugin
from .ert_script import ErtScript

logger = logging.getLogger(__name__)

ContentTypes = Union[Type[int], Type[bool], Type[float], Type[str]]


USE_NEW_PARSER_BY_DEFAULT = True

if "USE_OLD_ERT_PARSER" in os.environ and os.environ["USE_OLD_ERT_PARSER"] == "YES":
    USE_NEW_PARSER_BY_DEFAULT = False


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


def new_workflow_job_parser(file: str):
    schema = init_workflow_schema()
    return lark_parse(file, schema=schema)


class ErtScriptLoadFailure(ValueError):
    pass


class TupleWithOrigin(tuple):
    def __new__(cls, values, origin: str):
        obj = super().__new__(cls, values)
        obj.origin = origin
        return obj


@dataclass
class WorkflowJob:
    name: str
    internal: bool
    min_args: Optional[int]
    max_args: Optional[int]
    arg_types: List[ContentTypeEnum]
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
        # First find the number of args
        args_spec = content_dict.get(WorkflowJobKeys.ARG_TYPE, [])
        specified_highest_arg_index = (
            max(index for index, _ in args_spec) if len(args_spec) > 0 else -1
        )
        specified_max_args = content_dict.get("MAX_ARG")
        specified_min_args = content_dict.get("MIN_ARG")

        num_args = max(
            specified_highest_arg_index + 1,
            specified_max_args or 0,
            specified_min_args or 0,
        )

        arg_types_dict = dict()

        for i, type_as_string in args_spec:
            # make old tests pass temporarily, will use
            # SchemaItemType
            arg_types_dict[i] = WorkflowJob.stringToType(type_as_string)

        arg_types_list = [
            arg_types_dict.get(i, ContentTypeEnum.CONFIG_STRING)
            for i in range(num_args)
        ]
        return arg_types_list

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
    def _file_to_workflow_args(cls, config_file, name=None, use_new_parser=True):
        if not name:
            name = os.path.basename(config_file)

        if use_new_parser:
            new_content_dict = new_workflow_job_parser(config_file)

            new_types_list = cls._make_arg_types_list_new(new_content_dict)
            return (
                name,
                new_content_dict.get("INTERNAL"),
                new_content_dict.get("MIN_ARG"),
                new_content_dict.get("MAX_ARG"),
                new_types_list,
                new_content_dict.get("EXECUTABLE"),
                str(new_content_dict.get("SCRIPT"))
                if "SCRIPT" in new_content_dict
                else None,
            )
        else:
            old_content = _config_parser.parse(config_file)

            def optional_get(key):
                return old_content.getValue(key) if old_content.hasKey(key) else None

            max_arg = optional_get("MAX_ARG")

            old_arg_types_list = cls._make_arg_types_list(old_content, max_arg)
            return (
                name,
                optional_get("INTERNAL"),
                optional_get("MIN_ARG"),
                max_arg,
                old_arg_types_list,
                optional_get("EXECUTABLE"),
                optional_get("SCRIPT"),
            )

    @classmethod
    def fromFile(cls, config_file, name=None):
        if os.path.isfile(config_file) and os.access(config_file, os.R_OK):
            new = cls._file_to_workflow_args(config_file, name, use_new_parser=True)
            old = cls._file_to_workflow_args(config_file, name, use_new_parser=False)

            if new != old:

                def to_tuple_set(workflow_job_args, origin: str):
                    (
                        workflow_name,
                        internal,
                        min_arg,
                        max_arg,
                        arg_types_list,
                        executable,
                        script,
                    ) = workflow_job_args

                    return set(
                        [
                            TupleWithOrigin(("name", workflow_name), origin),
                            TupleWithOrigin(("internal", str(internal)), origin),
                            TupleWithOrigin(("min_arg", min_arg), origin),
                            TupleWithOrigin(("max_arg", max_arg), origin),
                            TupleWithOrigin(
                                ("arg_types_list", map(str, arg_types_list)), origin
                            ),
                            TupleWithOrigin(("executable", executable), origin),
                            TupleWithOrigin(("script", script), origin),
                        ]
                    )

                diff = to_tuple_set(new, "new") ^ to_tuple_set(old, "old")
                diff_formatted = [f"{x.origin}: {x}" for x in diff]

                logger.info(
                    "Old and new workflow job parser gave "
                    f"different results. {diff_formatted}"
                )

            return cls(*(new if USE_NEW_PARSER_BY_DEFAULT else old))

        else:
            raise ConfigValidationError(f"Could not open config_file:{config_file!r}")

    def isPlugin(self) -> bool:
        if self.ert_script is not None:
            return issubclass(self.ert_script, ErtPlugin)
        return False

    @classmethod
    def stringToType(cls, string: str) -> ContentTypeEnum:
        if string == "STRING":
            return ContentTypeEnum.CONFIG_STRING
        if string == "FLOAT":
            return ContentTypeEnum.CONFIG_FLOAT
        if string == "INT":
            return ContentTypeEnum.CONFIG_INT
        if string == "BOOL":
            return ContentTypeEnum.CONFIG_BOOL

        raise ValueError("Unrecognized content type")

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
