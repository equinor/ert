import logging
import os
import os.path
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ert._c_wrappers.config import ConfigParser
from ert._c_wrappers.util import SubstitutionList
from ert._clib.job_kw import type_from_kw
from ert.parsing import (
    ConfigValidationError,
    ExtJobKeys,
    SchemaItemType,
    init_ext_job_schema,
    lark_parse,
)

from ...parsing.error_info import ErrorInfo
from .parse_arg_types_list import parse_arg_types_list

logger = logging.getLogger(__name__)


@dataclass
class ExtJob:
    name: str
    executable: str
    stdin_file: Optional[str] = None
    stdout_file: Optional[str] = None
    stderr_file: Optional[str] = None
    start_file: Optional[str] = None
    target_file: Optional[str] = None
    error_file: Optional[str] = None
    max_running: Optional[int] = None
    max_running_minutes: Optional[int] = None
    min_arg: Optional[int] = None
    max_arg: Optional[int] = None
    arglist: List[str] = field(default_factory=list)
    arg_types: List[SchemaItemType] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    exec_env: Dict[str, str] = field(default_factory=dict)
    default_mapping: Dict[str, str] = field(default_factory=dict)
    private_args: SubstitutionList = field(default_factory=SubstitutionList)
    help_text: str = ""

    @staticmethod
    def _resolve_executable(executable, name, config_file_location):
        """
        :returns: The resolved path to the executable

        :raises: ConfigValidationError if the executable cannot be found
            or we don't have permissions to execute.
        """
        # PS: This operation has surprising behavior but is kept this way for
        # backwards compatability
        if not os.path.isabs(executable):
            path_to_executable = os.path.abspath(
                os.path.join(config_file_location, executable)
            )
        else:
            path_to_executable = executable

        resolved = None
        if os.path.exists(path_to_executable):
            resolved = path_to_executable
        elif not os.path.isabs(executable):
            # look for system installed executable
            resolved = shutil.which(executable)

        if resolved is None:
            raise ConfigValidationError(
                config_file=config_file_location,
                errors=f"Could not find executable {executable!r} for job {name!r}",
            )

        if not os.access(resolved, os.X_OK):  # is not executable
            raise ConfigValidationError(
                config_file=config_file_location,
                errors=f"ExtJob {name!r} with executable"
                f" {resolved!r} does not have execute permissions",
            )

        if os.path.isdir(resolved):
            raise ConfigValidationError(
                config_file=config_file_location,
                errors=f"ExtJob {name!r} has executable set to directory {resolved!r}",
            )

        return resolved

    _int_keywords = ["MAX_RUNNING", "MAX_RUNNING_MINUTES", "MIN_ARG", "MAX_ARG"]
    _str_keywords = [
        "STDIN",
        "STDOUT",
        "STDERR",
        "START_FILE",
        "TARGET_FILE",
        "ERROR_FILE",
        "START_FILE",
    ]
    default_env = {
        "_ERT_ITERATION_NUMBER": "<ITER>",
        "_ERT_REALIZATION_NUMBER": "<IENS>",
        "_ERT_RUNPATH": "<RUNPATH>",
    }

    @classmethod
    def _parse_config_file(cls, config_file: str):
        parser = ConfigParser()
        for int_key in cls._int_keywords:
            parser.add(int_key, value_type=SchemaItemType.INT).set_argc_minmax(1, 1)
        for path_key in cls._str_keywords:
            parser.add(path_key).set_argc_minmax(1, 1)

        parser.add("EXECUTABLE", required=True).set_argc_minmax(1, 1)
        parser.add("ENV").set_argc_minmax(2, 2)
        parser.add("EXEC_ENV").set_argc_minmax(2, 2)
        parser.add("DEFAULT").set_argc_minmax(2, 2)
        parser.add("ARGLIST").set_argc_minmax(1, -1)
        arg_type_schema = parser.add("ARG_TYPE")
        arg_type_schema.set_argc_minmax(2, 2)
        arg_type_schema.iset_type(0, SchemaItemType.INT)

        return parser.parse(
            config_file,
        )

    @classmethod
    def _read_str_keywords(cls, config_content):
        result = {}

        def might_set_value_none(keyword, key):
            value = config_content.getValue(keyword)
            if value == "null":
                value = None
            result[key] = value

        for key in cls._str_keywords:
            if config_content.hasKey(key):
                if key in ("STDIN", "STDOUT", "STDERR"):
                    might_set_value_none(key, key.lower() + "_file")
                else:
                    might_set_value_none(key, key.lower())
        return result

    @classmethod
    def _read_int_keywords(cls, config_content):
        result = {}
        for key in cls._int_keywords:
            if config_content.hasKey(key):
                value = config_content.getValue(key)
                if value > 0:
                    # less than or equal to 0 in the config is equivalent to
                    # setting None (backwards compatability)
                    result[key.lower()] = value
        return result

    @staticmethod
    def _make_arg_types_list(
        config_content, max_arg: Optional[int]
    ) -> List[SchemaItemType]:
        arg_types_dict = defaultdict(lambda: SchemaItemType.STRING)
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
    def from_config_file_with_new_parser(
        cls, config_file: str, name: Optional[str] = None
    ):
        schema = init_ext_job_schema()

        try:
            content_dict = lark_parse(file=config_file, schema=schema)

            specified_arg_types: List[Tuple[int, str]] = content_dict.get(
                ExtJobKeys.ARG_TYPE, []
            )  # type: ignore

            specified_max_args: int = content_dict.get("MAX_ARG", 0)  # type: ignore
            specified_min_args: int = content_dict.get("MIN_ARG", 0)  # type: ignore

            arg_types_list = parse_arg_types_list(
                specified_arg_types, specified_max_args, specified_min_args
            )

            # We unescape backslash here to keep backwards compatability ie. If
            # the arglist contains a '\n' we interpret it as a newline.
            arglist = [
                s.encode("utf-8", "backslashreplace").decode("unicode_escape")
                for s in content_dict.get("ARGLIST", [])
            ]

            environment = {k: v for [k, v] in content_dict.get("ENV", [])}
            exec_env = {k: v for [k, v] in content_dict.get("EXEC_ENV", [])}
            default_mapping = {k: v for [k, v] in content_dict.get("DEFAULT", [])}

            environment.update(cls.default_env)

            for handle in ["STDOUT", "STDERR", "STDIN"]:
                if content_dict.get(handle) == "null":
                    content_dict[handle] = None

            stdout_file = content_dict.get("STDOUT", f"{name}.stdout")
            stderr_file = content_dict.get("STDERR", f"{name}.stderr")

            return cls(
                name=name,
                executable=content_dict.get("EXECUTABLE"),
                stdin_file=content_dict.get("STDIN"),
                stdout_file=stdout_file,
                stderr_file=stderr_file,
                start_file=content_dict.get("START_FILE"),
                target_file=content_dict.get("TARGET_FILE"),
                error_file=content_dict.get("ERROR_FILE"),
                max_running=content_dict.get("MAX_RUNNING"),
                max_running_minutes=content_dict.get("MAX_RUNNING_MINUTES"),
                min_arg=content_dict.get("MIN_ARG"),
                max_arg=content_dict.get("MAX_ARG"),
                arglist=arglist,
                arg_types=arg_types_list,
                environment=environment,
                exec_env=exec_env,
                default_mapping=default_mapping,
                help_text=content_dict.get("HELP_TEXT"),
            )
        except IOError as err:
            raise ConfigValidationError.from_info(
                ErrorInfo(message=str(err), filename=name)
            )

    @classmethod
    def from_config_file_with_old_parser(
        cls, config_file: str, name: Optional[str] = None
    ):
        try:
            config_content = cls._parse_config_file(config_file)
        except ConfigValidationError as conf_err:
            with open(config_file, encoding="utf-8") as f:
                if "PORTABLE_EXE " in f.read():
                    err_msg = (
                        '"PORTABLE_EXE" key is deprecated,'
                        ' please replace with "EXECUTABLE" in'
                    )
                    raise ConfigValidationError.from_collected(
                        [conf_err, ConfigValidationError(err_msg, config_file)]
                    ) from None
            raise conf_err from None

        except IOError as err:
            raise ConfigValidationError(
                config_file=config_file,
                errors=f"Could not open job config file {config_file!r}",
            ) from err

        logger.info(
            "Content of job config %s: %s",
            name,
            Path(config_file).read_text(encoding="utf-8"),
        )
        content_dict = {}

        content_dict.update(**cls._read_str_keywords(config_content))
        content_dict.update(**cls._read_int_keywords(config_content))

        content_dict["executable"] = config_content.getValue("EXECUTABLE")
        if config_content.hasKey("ARGLIST"):
            # We unescape backslash here to keep backwards compatability ie. If
            # the arglist contains a '\n' we interpret it as a newline.
            content_dict["arglist"] = [
                s.encode("utf-8", "backslashreplace").decode("unicode_escape")
                for s in config_content["ARGLIST"][-1]
            ]

        content_dict["arg_types"] = cls._make_arg_types_list(
            config_content,
            content_dict["max_arg"]
            if "max_arg" in content_dict and content_dict["max_arg"] > 0
            else None,
        )

        def set_env(key, keyword):
            content_dict[key] = {}
            if config_content.hasKey(keyword):
                for env in config_content[keyword]:
                    if len(env) > 1:
                        content_dict[key][env[0]] = env[1]
                    else:
                        content_dict[key][env[0]] = None

        set_env("environment", "ENV")
        set_env("exec_env", "EXEC_ENV")
        # Add default run information to job environment vars
        content_dict["environment"].update(cls.default_env)

        content_dict["default_mapping"] = {}
        if config_content.hasKey("DEFAULT"):
            for key, value in config_content["DEFAULT"]:
                content_dict["default_mapping"][key] = value

        content_dict["executable"] = ExtJob._resolve_executable(
            content_dict["executable"], name, os.path.dirname(config_file)
        )

        # The default for stdout_file and stdin_file is
        # {name}.std{out/err}
        for handle in ("stdout", "stderr"):
            if handle + "_file" not in content_dict:
                content_dict[handle + "_file"] = name + "." + handle

        return cls(
            name,
            **content_dict,
        )

    @classmethod
    def from_config_file(
        cls, config_file: str, name: Optional[str] = None, use_new_parser: bool = True
    ):
        if name is None:
            name = os.path.basename(config_file)

        if use_new_parser:
            return cls.from_config_file_with_new_parser(config_file, name)
        else:
            return cls.from_config_file_with_old_parser(config_file, name)
