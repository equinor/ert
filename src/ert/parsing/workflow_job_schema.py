from typing import Any, Dict

from . import ConfigValidationError
from .config_schema_item import SchemaItem
from .schema_dict import SchemaItemDict
from .schema_item_type import SchemaItemType
from .workflow_job_keywords import WorkflowJobKeys


def min_arg_keyword() -> SchemaItem:
    return SchemaItem(
        kw=WorkflowJobKeys.MIN_ARG,
        required_set=False,
        argc_min=1,
        argc_max=1,
        type_map=[SchemaItemType.INT],
    )


def max_arg_keyword() -> SchemaItem:
    return SchemaItem(
        kw=WorkflowJobKeys.MAX_ARG,
        required_set=False,
        argc_min=1,
        argc_max=1,
        type_map=[SchemaItemType.INT],
    )


def arg_type_keyword() -> SchemaItem:
    return SchemaItem(
        kw=WorkflowJobKeys.ARG_TYPE,
        required_set=False,
        argc_min=2,
        argc_max=2,
        type_map=[SchemaItemType.INT, SchemaItemType.STRING],
        multi_occurrence=True,
    )


def arglist_keyword() -> SchemaItem:
    return SchemaItem(
        kw=WorkflowJobKeys.ARGLIST,
        required_set=False,
        argc_min=1,
        argc_max=1,
        join_after=1,
    )


def executable_keyword() -> SchemaItem:
    return SchemaItem(
        kw=WorkflowJobKeys.EXECUTABLE,
        required_set=False,
        argc_min=1,
        argc_max=1,
        type_map=[SchemaItemType.EXECUTABLE],
    )


def script_keyword() -> SchemaItem:
    return SchemaItem(
        kw=WorkflowJobKeys.SCRIPT,
        required_set=False,
        argc_min=1,
        argc_max=1,
        type_map=[SchemaItemType.PATH],
    )


def internal_keyword() -> SchemaItem:
    return SchemaItem(
        kw=WorkflowJobKeys.INTERNAL,
        required_set=False,
        type_map=[SchemaItemType.BOOL],
    )


class WorkflowJobSchemaDict(SchemaItemDict):
    def check_required(self, config_dict: Dict[str, Any], filename: str):
        super().check_required(config_dict, filename)

        if "MIN_ARG" in config_dict and "MAX_ARG" in config_dict:
            min_arg: int = config_dict["MIN_ARG"]
            max_arg: int = config_dict["MAX_ARG"]

            assert isinstance(min_arg, int)
            assert isinstance(max_arg, int)

            if max_arg < 0:
                raise ConfigValidationError("specified MAX_ARG must be at least 0.")

            if min_arg > max_arg:
                raise ConfigValidationError(
                    f"MIN_ARG ({min_arg}) must be lesser than MAX_ARG ({max_arg})"
                )


def init_workflow_schema() -> SchemaItemDict:
    schema = WorkflowJobSchemaDict()
    for item in [
        executable_keyword(),
        arglist_keyword(),
        script_keyword(),
        internal_keyword(),
        min_arg_keyword(),
        max_arg_keyword(),
        arg_type_keyword(),
    ]:
        schema[item.kw] = item

    return schema
