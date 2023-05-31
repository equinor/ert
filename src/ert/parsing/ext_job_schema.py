from typing import List

from .config_schema_item import SchemaItem, SchemaItemType
from .ext_job_keywords import ExtJobKeys
from .schema_dict import SchemaItemDict


def executable_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.EXECUTABLE,
        required_set=True,
        argc_min=1,
        argc_max=1,
        type_map=[SchemaItemType.EXECUTABLE],
    )


def stdin_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.STDIN,
        type_map=[SchemaItemType.STRING],
        argc_min=1,
        argc_max=1,
        required_set=False,
    )


def stdout_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.STDOUT,
        type_map=[SchemaItemType.STRING],
        argc_min=1,
        argc_max=1,
        required_set=False,
    )


def stderr_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.STDERR,
        type_map=[SchemaItemType.STRING],
        argc_min=1,
        argc_max=1,
        required_set=False,
    )


def start_file_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.START_FILE,
        argc_min=1,
        argc_max=1,
        type_map=[SchemaItemType.STRING],
    )


def target_file_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.TARGET_FILE,
        argc_min=1,
        argc_max=1,
        type_map=[SchemaItemType.STRING],
    )


def error_file_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.ERROR_FILE,
        argc_min=1,
        argc_max=1,
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
        argc_min=1,
        argc_max=1,
        required_set=False,
    )


def max_arg_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ExtJobKeys.MAX_ARG,
        type_map=[SchemaItemType.INT],
        argc_min=1,
        argc_max=1,
        required_set=False,
    )


def arglist_keyword() -> SchemaItem:
    return SchemaItem(kw=ExtJobKeys.ARGLIST, argc_min=1, argc_max=-1, substitute_from=0)


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


def portable_exe_keyword() -> SchemaItem:
    return SchemaItem(
        kw="PORTABLE_EXE",
        deprecated=True,
        deprecate_msg='"PORTABLE_EXE" key is deprecated, '
        'please replace with "EXECUTABLE"',
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
    portable_exe_keyword(),
]


def init_ext_job_schema() -> SchemaItemDict:
    schema = SchemaItemDict()

    for item in ext_job_schema_items:
        schema[item.kw] = item

    return schema
