from typing import List

from .config_dict import ConfigDict
from .config_schema_item import SchemaItem
from .deprecation_info import DeprecationInfo
from .ext_job_keywords import ExtJobKeys
from .schema_dict import SchemaItemDict
from .schema_item_type import SchemaItemType


def executable_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.EXECUTABLE,
        required_set=True,
        type_map=[SchemaItemType.EXECUTABLE],
    )


def stdin_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.STDIN,
        type_map=[SchemaItemType.STRING],
        required_set=False,
    )


def stdout_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.STDOUT,
        type_map=[SchemaItemType.STRING],
        required_set=False,
    )


def stderr_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.STDERR,
        type_map=[SchemaItemType.STRING],
        required_set=False,
    )


def start_file_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.START_FILE,
        type_map=[SchemaItemType.STRING],
    )


def target_file_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.TARGET_FILE,
        type_map=[SchemaItemType.STRING],
    )


def error_file_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.ERROR_FILE,
        type_map=[SchemaItemType.STRING],
    )


def max_running_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.MAX_RUNNING, type_map=[SchemaItemType.INT], required_set=False
    )


def max_running_minutes_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.MAX_RUNNING_MINUTES,
        type_map=[SchemaItemType.INT],
        required_set=False,
    )


def min_arg_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.MIN_ARG,
        type_map=[SchemaItemType.INT],
        required_set=False,
    )


def max_arg_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.MAX_ARG,
        type_map=[SchemaItemType.INT],
        required_set=False,
    )


def arglist_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.ARGLIST,
        argc_max=None,
    )


def arg_type_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.ARG_TYPE,
        argc_min=2,
        argc_max=2,
        type_map=[SchemaItemType.INT, SchemaItemType.STRING],
        multi_occurrence=True,
    )


def env_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.ENV,
        argc_min=2,
        argc_max=2,
        multi_occurrence=True,
        type_map=[SchemaItemType.STRING, SchemaItemType.STRING],
    )


def exec_env_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.EXEC_ENV,
        argc_min=2,
        argc_max=2,
        multi_occurrence=True,
        type_map=[SchemaItemType.STRING, SchemaItemType.STRING],
    )


def default_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.DEFAULT,
        argc_min=2,
        argc_max=2,
        multi_occurrence=True,
        type_map=[SchemaItemType.STRING, SchemaItemType.STRING],
    )


ext_job_schema_items: List[SchemaItem] = [
    executable_keyword(),
    stdin_keyword(),
    stdout_keyword(),
    stderr_keyword(),
    start_file_keyword(),
    target_file_keyword(),
    error_file_keyword(),
    max_running_keyword(),
    max_running_minutes_keyword(),
    min_arg_keyword(),
    max_arg_keyword(),
    arglist_keyword(),
    default_keyword(),  # Default values for args
    arg_type_keyword(),
    env_keyword(),
    exec_env_keyword(),
]

ext_job_deprecations: List[DeprecationInfo] = [
    DeprecationInfo(
        keyword="PORTABLE_EXE",
        message='"PORTABLE_EXE" key is deprecated, please replace with "EXECUTABLE"',
    )
]


class ExtJobSchemaItemDict(SchemaItemDict):
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


def init_ext_job_schema() -> ExtJobSchemaItemDict:
    schema = ExtJobSchemaItemDict()

    for item in ext_job_schema_items:
        schema[item.kw] = item

    schema.add_deprecations(ext_job_deprecations)
    return schema
