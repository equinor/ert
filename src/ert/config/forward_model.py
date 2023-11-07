import logging
import os
import os.path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, no_type_check

from ert.substitution_list import SubstitutionList

from .parse_arg_types_list import parse_arg_types_list
from .parsing import (
    ConfigValidationError,
    ForwardModelKeys,
    SchemaItemType,
    init_forward_model_schema,
    lark_parse,
)

logger = logging.getLogger(__name__)


@dataclass
class ForwardModel:
    name: str
    executable: str
    stdin_file: Optional[str] = None
    stdout_file: Optional[str] = None
    stderr_file: Optional[str] = None
    start_file: Optional[str] = None
    target_file: Optional[str] = None
    error_file: Optional[str] = None
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

    default_env = {
        "_ERT_ITERATION_NUMBER": "<ITER>",
        "_ERT_REALIZATION_NUMBER": "<IENS>",
        "_ERT_RUNPATH": "<RUNPATH>",
    }

    @no_type_check
    @classmethod
    def from_config_file(
        cls, config_file: str, name: Optional[str] = None
    ) -> "ForwardModel":
        if name is None:
            name = os.path.basename(config_file)

        schema = init_forward_model_schema()

        try:
            content_dict = lark_parse(file=config_file, schema=schema, pre_defines=[])

            specified_arg_types: List[Tuple[int, str]] = content_dict.get(
                ForwardModelKeys.ARG_TYPE, []
            )

            specified_max_args: int = content_dict.get("MAX_ARG", 0)
            specified_min_args: int = content_dict.get("MIN_ARG", 0)

            arg_types_list = parse_arg_types_list(
                specified_arg_types, specified_min_args, specified_max_args
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
                max_running_minutes=content_dict.get("MAX_RUNNING_MINUTES"),
                min_arg=content_dict.get("MIN_ARG"),
                max_arg=content_dict.get("MAX_ARG"),
                arglist=arglist,
                arg_types=arg_types_list,
                environment=environment,
                exec_env=exec_env,
                default_mapping=default_mapping,
                help_text=content_dict.get("HELP_TEXT", ""),
            )
        except IOError as err:
            raise ConfigValidationError.with_context(str(err), config_file) from err
