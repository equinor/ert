from typing import List

from .config_dict import ConfigDict
from .config_schema_item import SchemaItem
from .deprecation_info import DeprecationInfo
from .forward_model_keywords import ForwardModelStepKeys
from .schema_dict import SchemaItemDict
from .schema_item_type import SchemaItemType


def executable_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ForwardModelStepKeys.EXECUTABLE,
        required_set=True,
        type_map=[SchemaItemType.EXECUTABLE],
    )


def stdin_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ForwardModelStepKeys.STDIN,
        type_map=[SchemaItemType.STRING],
        required_set=False,
    )


def stdout_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ForwardModelStepKeys.STDOUT,
        type_map=[SchemaItemType.STRING],
        required_set=False,
    )


def stderr_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ForwardModelStepKeys.STDERR,
        type_map=[SchemaItemType.STRING],
        required_set=False,
    )


def start_file_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ForwardModelStepKeys.START_FILE,
        type_map=[SchemaItemType.STRING],
    )


def target_file_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ForwardModelStepKeys.TARGET_FILE,
        type_map=[SchemaItemType.STRING],
    )


def error_file_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ForwardModelStepKeys.ERROR_FILE,
        type_map=[SchemaItemType.STRING],
    )


def max_running_minutes_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ForwardModelStepKeys.MAX_RUNNING_MINUTES,
        type_map=[SchemaItemType.INT],
        required_set=False,
    )


def min_arg_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ForwardModelStepKeys.MIN_ARG,
        type_map=[SchemaItemType.INT],
        required_set=False,
    )


def max_arg_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ForwardModelStepKeys.MAX_ARG,
        type_map=[SchemaItemType.INT],
        required_set=False,
    )


def arglist_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ForwardModelStepKeys.ARGLIST,
        argc_max=None,
    )


def required_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ForwardModelStepKeys.REQUIRED,
        argc_max=None,
    )


def arg_type_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ForwardModelStepKeys.ARG_TYPE,
        argc_min=2,
        argc_max=2,
        type_map=[SchemaItemType.INT, SchemaItemType.STRING],
        multi_occurrence=True,
    )


def env_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ForwardModelStepKeys.ENV,
        argc_min=2,
        argc_max=2,
        multi_occurrence=True,
        type_map=[SchemaItemType.STRING, SchemaItemType.STRING],
    )


def exec_env_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ForwardModelStepKeys.EXEC_ENV,
        argc_min=2,
        argc_max=2,
        multi_occurrence=True,
        type_map=[SchemaItemType.STRING, SchemaItemType.STRING],
    )


def default_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ForwardModelStepKeys.DEFAULT,
        argc_min=2,
        argc_max=2,
        multi_occurrence=True,
        type_map=[SchemaItemType.STRING, SchemaItemType.STRING],
    )


forward_model_schema_items: List[SchemaItem] = [
    executable_keyword(),
    stdin_keyword(),
    stdout_keyword(),
    stderr_keyword(),
    start_file_keyword(),
    target_file_keyword(),
    error_file_keyword(),
    max_running_minutes_keyword(),
    min_arg_keyword(),
    max_arg_keyword(),
    arglist_keyword(),
    required_keyword(),
    default_keyword(),  # Default values for args
    arg_type_keyword(),
    env_keyword(),
    exec_env_keyword(),
]

forward_model_deprecations: List[DeprecationInfo] = [
    DeprecationInfo(
        keyword="PORTABLE_EXE",
        message='"PORTABLE_EXE" key is deprecated, please replace with "EXECUTABLE"',
    ),
    DeprecationInfo(
        keyword="MAX_RUNNING",
        message='"MAX_RUNNING" in a forward model configuration is not doing anything. '
        "You can safely remove it.",
    ),
]


class ForwardModelSchemaItemDict(SchemaItemDict):
    def check_required(
        self,
        config_dict: ConfigDict,
        filename: str,
    ) -> None:
        self.search_for_deprecated_keyword_usages(
            config_dict=config_dict,
            filename=filename,
        )
        self.search_for_unset_required_keywords(
            config_dict=config_dict, filename=filename
        )


def init_forward_model_schema() -> ForwardModelSchemaItemDict:
    schema = ForwardModelSchemaItemDict()

    for item in forward_model_schema_items:
        schema[item.kw] = item

    schema.add_deprecations(forward_model_deprecations)
    return schema
