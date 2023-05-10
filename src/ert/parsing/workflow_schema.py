from typing import Dict


from .config_keywords import (
    ConfigKeys,
    RunModes,
    QueueOptions,
)
from .schema_item_type import SchemaItemType
from .config_schema_item import (
    SchemaItem,
    existing_path_keyword,
    single_arg_keyword,
    path_keyword,
    string_keyword,
    int_keyword,
    float_keyword,
)
from .workflow_keywords import WorkflowJobKeys


def min_arg_keyword() -> SchemaItem:
    return SchemaItem(
        kw=WorkflowJobKeys.MIN_ARG,
        required_set=True,
        argc_min=1,
        argc_max=1,
        type_map=[SchemaItemType.CONFIG_INT],
    )


def max_arg_keyword() -> SchemaItem:
    return SchemaItem(
        kw=WorkflowJobKeys.MAX_ARG,
        required_set=False,
        argc_min=1,
        argc_max=1,
        type_map=[SchemaItemType.CONFIG_INT],
    )


def arg_type_keyword() -> SchemaItem:
    return SchemaItem(
        kw=WorkflowJobKeys.ARG_TYPE,
        required_set=False,
        argc_min=2,
        argc_max=2,
        type_map=[SchemaItemType.CONFIG_INT, SchemaItemType.CONFIG_STRING],
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
        type_map=[SchemaItemType.CONFIG_EXECUTABLE],
    )


def script_keyword() -> SchemaItem:
    return SchemaItem(
        kw=WorkflowJobKeys.SCRIPT,
        required_set=False,
        argc_min=1,
        argc_max=1,
        type_map=[SchemaItemType.CONFIG_PATH],
    )


def internal_keyword() -> SchemaItem:
    return SchemaItem(
        kw=WorkflowJobKeys.INTERNAL,
        required_set=False,
        type_map=[SchemaItemType.CONFIG_BOOL],
    )


def init_workflow_schema() -> Dict[str, SchemaItem]:
    schema = {}
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
