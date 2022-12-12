import os
import os.path
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ert._c_wrappers.config import ConfigParser, ConfigValidationError, ContentTypeEnum
from ert._c_wrappers.util import SubstitutionList
from ert._clib.job_kw import type_from_kw


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
    arg_types: List[ContentTypeEnum] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    exec_env: Dict[str, str] = field(default_factory=dict)
    default_mapping: Dict[str, str] = field(default_factory=dict)
    private_args: SubstitutionList = field(default_factory=SubstitutionList)
    define_args: SubstitutionList = field(default_factory=SubstitutionList)
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
                errors=[f"Could not find executable {executable} for job {name}"],
            )

        if not os.access(resolved, os.X_OK):  # is not executable
            raise ConfigValidationError(
                config_file=config_file_location,
                errors=[
                    f"ExtJob {name} with executable"
                    f" {resolved} does not have execute permissions"
                ],
            )

        if os.path.isdir(resolved):
            raise ConfigValidationError(
                config_file=config_file_location,
                errors=[f"ExtJob {name} has executable set to directory {resolved}"],
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

    @classmethod
    def _parse_config_file(cls, config_file: str):
        parser = ConfigParser()
        for int_key in cls._int_keywords:
            parser.add(int_key, value_type=ContentTypeEnum.CONFIG_INT).set_argc_minmax(
                1, 1
            )
        for path_key in cls._str_keywords:
            parser.add(path_key).set_argc_minmax(1, 1)

        parser.add("EXECUTABLE", required=True).set_argc_minmax(1, 1)
        parser.add("ENV").set_argc_minmax(1, 2)
        parser.add("EXEC_ENV").set_argc_minmax(1, 2)
        parser.add("DEFAULT").set_argc_minmax(2, 2)
        parser.add("ARGLIST").set_argc_minmax(1, -1)
        arg_type_schema = parser.add("ARG_TYPE")
        arg_type_schema.set_argc_minmax(2, 2)
        arg_type_schema.iset_type(0, ContentTypeEnum.CONFIG_INT)

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
    def from_config_file(cls, config_file: str, name: Optional[str] = None):
        if name is None:
            name = os.path.basename(config_file)

        try:
            config_content = cls._parse_config_file(config_file)
        except IOError as err:
            raise ConfigValidationError(
                config_file=config_file,
                errors=f"Could not open job config file {config_file}",
            ) from err

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
