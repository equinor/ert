import os
import shutil
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Union

from pydantic import BaseModel

from . import ConfigValidationError
from .lark_parser_error_info import ErrorInfo
from .lark_parser_file_context_token import FileContextToken
from .lark_parser_primitive_tokens import (
    BoolToken,
    FloatToken,
    IntToken,
    PrimitiveWithContext,
    StringToken,
)
from .lark_parser_types import Instruction

# These keys are used as options in KEY:VALUE statements
BASE_SURFACE_KEY = "BASE_SURFACE"
DEFINE_KEY = "DEFINE"
FORWARD_INIT_KEY = "FORWARD_INIT"
GENERAL_KEY = "GENERAL"
INCLUDE_KEY = "INCLUDE"
INIT_FILES_KEY = "INIT_FILES"
INIT_TRANSFORM_KEY = "INIT_TRANSFORM"
INPUT_FORMAT_KEY = "INPUT_FORMAT"
INPUT_TRANSFORM_KEY = "INPUT_TRANSFORM"
MAX_KEY = "MAX"
MIN_KEY = "MIN"
OUTPUT_FILE_KEY = "OUTPUT_FILE"
OUTPUT_TRANSFORM_KEY = "OUTPUT_TRANSFORM"
PARAMETER_KEY = "PARAMETER"
REPORT_STEPS_KEY = "REPORT_STEPS"
RESULT_FILE_KEY = "RESULT_FILE"
TEMPLATE_KEY = "TEMPLATE"
PRED_KEY = "PRED_KEY"

ANALYSIS_COPY_KEY = "ANALYSIS_COPY"
ANALYSIS_SET_VAR_KEY = "ANALYSIS_SET_VAR"
ANALYSIS_SELECT_KEY = "ANALYSIS_SELECT"
DATA_ROOT_KEY = "DATA_ROOT"
DATA_FILE_KEY = "DATA_FILE"
DATA_KW_KEY = "DATA_KW"
ECLBASE_KEY = "ECLBASE"
ENKF_ALPHA_KEY = "ENKF_ALPHA"
ENSPATH_KEY = "ENSPATH"
ITER_CASE_KEY = "ITER_CASE"
ITER_COUNT_KEY = "ITER_COUNT"
ITER_RETRY_COUNT_KEY = "ITER_RETRY_COUNT"
FIELD_KEY = "FIELD"
FORWARD_MODEL_KEY = "FORWARD_MODEL"
GEN_DATA_KEY = "GEN_DATA"
GEN_KW_KEY = "GEN_KW"
GEN_KW_TAG_FORMAT_KEY = "GEN_KW_TAG_FORMAT"
GEN_KW_EXPORT_NAME_KEY = "GEN_KW_EXPORT_NAME"
GRID_KEY = "GRID"
HISTORY_SOURCE_KEY = "HISTORY_SOURCE"
INSTALL_JOB_KEY = "INSTALL_JOB"
INSTALL_JOB_DIRECTORY_KEY = "INSTALL_JOB_DIRECTORY"
JOB_SCRIPT_KEY = "JOB_SCRIPT"
JOBNAME_KEY = "JOBNAME"
LICENSE_PATH_KEY = "LICENSE_PATH"
MAX_SUBMIT_KEY = "MAX_SUBMIT"
NUM_REALIZATIONS_KEY = "NUM_REALIZATIONS"
MIN_REALIZATIONS_KEY = "MIN_REALIZATIONS"
OBS_CONFIG_KEY = "OBS_CONFIG"
QUEUE_SYSTEM_KEY = "QUEUE_SYSTEM"
QUEUE_OPTION_KEY = "QUEUE_OPTION"
HOOK_WORKFLOW_KEY = "HOOK_WORKFLOW"
REFCASE_KEY = "REFCASE"
RUNMODE_KEY = "RUNMODE"
RUNPATH_FILE_KEY = "RUNPATH_FILE"
RUNPATH_KEY = "RUNPATH"
RUN_TEMPLATE_KEY = "RUN_TEMPLATE"
SCHEDULE_PREDICTION_FILE_KEY = "SCHEDULE_PREDICTION_FILE"
SETENV_KEY = "SETENV"
SIMULATION_JOB_KEY = "SIMULATION_JOB"
STD_CUTOFF_KEY = "STD_CUTOFF"
SUMMARY_KEY = "SUMMARY"
SURFACE_KEY = "SURFACE"
UPDATE_LOG_PATH_KEY = "UPDATE_LOG_PATH"
UPDATE_PATH_KEY = "UPDATE_PATH"
RANDOM_SEED_KEY = "RANDOM_SEED"
WORKFLOW_JOB_DIRECTORY_KEY = "WORKFLOW_JOB_DIRECTORY"
LOAD_WORKFLOW_KEY = "LOAD_WORKFLOW"
LOAD_WORKFLOW_JOB_KEY = "LOAD_WORKFLOW_JOB"
RUN_MODE_PRE_SIMULATION_NAME = "PRE_SIMULATION"
RUN_MODE_POST_SIMULATION_NAME = "POST_SIMULATION"
RUN_MODE_PRE_UPDATE_NAME = "PRE_UPDATE"
RUN_MODE_POST_UPDATE_NAME = "POST_UPDATE"
RUN_MODE_PRE_FIRST_UPDATE_NAME = "PRE_FIRST_UPDATE"
STOP_LONG_RUNNING_KEY = "STOP_LONG_RUNNING"
MAX_RUNTIME_KEY = "MAX_RUNTIME"
TIME_MAP_KEY = "TIME_MAP"
UPDATE_SETTING_KEY = "UPDATE_SETTINGS"
NUM_CPU_KEY = "NUM_CPU"

CONFIG_DIRECTORY_KEY = "CONFIG_DIRECTORY"

SLURM_SBATCH_OPTION = "SBATCH"
SLURM_SCANCEL_OPTION = "SCANCEL"
SLURM_SCONTROL_OPTION = "SCONTROL"
SLURM_SQUEUE_OPTION = "SQUEUE"
SLURM_PARTITION_OPTION = "PARTITION"
SLURM_SQUEUE_TIMEOUT_OPTION = "SQUEUE_TIMEOUT"

# Observe that the SLURM_MAX_RUNTIME_OPTION expects a time limit in seconds,
# whereas slurm uses a time limit in minutes
SLURM_MAX_RUNTIME_OPTION = "MAX_RUNTIME"
SLURM_MEMORY_OPTION = "MEMORY"
SLURM_MEMORY_PER_CPU_OPTION = "MEMORY_PER_CPU"

# For the EXCLUDE and INCLUDE host options the slurm driver
# maintains an internal list of hostnames, and the option can be called
# repeatedly. It is possible to add multiple hosts separated by space or comma
# in one option call:
#
# QUEUE_OPTION SLURM EXCLUDE_HOST host1,host2,host3
# QUEUE_OPTION SLURM EXCLUDE_HOST host5 host6,host7
SLURM_EXCLUDE_HOST_OPTION = "EXCLUDE_HOST"
SLURM_INCLUDE_HOST_OPTION = "INCLUDE_HOST"

CONFIG_DEFAULT_ARG_MAX = -1
CONFIG_DEFAULT_ARG_MIN = -1

ALIASES = {NUM_REALIZATIONS_KEY: ["NUM_REALISATIONS"]}


class SchemaType(Enum):
    CONFIG_STRING = (1,)
    CONFIG_INT = (2,)
    CONFIG_FLOAT = (4,)
    CONFIG_PATH = (8,)
    CONFIG_EXISTING_PATH = (16,)
    CONFIG_BOOL = (32,)
    CONFIG_CONFIG = (64,)
    CONFIG_BYTESIZE = (128,)
    CONFIG_EXECUTABLE = (256,)
    CONFIG_ISODATE = (512,)
    CONFIG_INVALID = (1024,)
    CONFIG_RUNTIME_INT = (2048,)
    CONFIG_RUNTIME_FILE = 4096


class SchemaItem(BaseModel):
    # The kw which identifies this item
    kw: str

    # The minimum number of arguments: -1 means no lower limit.
    argc_min: int = 1
    # The maximum number of arguments: -1 means no upper limit
    argc_max: int = 1
    # A list of types for the items. Set along with argc_minmax()
    type_map: List[Optional[SchemaType]] = []
    # A list of item's which must also be set (if this item is set). (can be NULL)
    required_children: List[str] = []
    # children that are required if certain values occur as argument to the item
    # Should environment variables like $HOME be expanded?
    deprecated: bool = False
    deprecate_msg: str = ""
    # if positive, arguments after this count will be concatenated with a " " between
    join_after: int = -1
    # if true, will accumulate many values set for key, otherwise each entry will
    # overwrite any previous value set
    multi_occurrence: bool = False
    expand_envvar: bool = True
    # Index of tokens to do substitution from until end,
    # 0 means no substitution, as keyword is never substituted
    substitute_from: int = 1
    required_set: bool = False
    required_children_value: Mapping[str, List[str]] = {}
    # Allowed values for arguments, if empty, all values allowed
    common_selection_set: List[str] = []
    # Allowed values for specific arguments, if no entry, all values allowed
    indexed_selection_set: Mapping[int, List[str]] = {}

    def _is_in_allowed_values_for_kw(
        self, token: "FileContextToken", index: int
    ) -> bool:
        return not (
            index in self.indexed_selection_set
            and token not in self.indexed_selection_set[index]
        )

    def token_to_value_with_context(
        self, token: FileContextToken, index: int, keyword: FileContextToken
    ) -> Optional[PrimitiveWithContext]:
        """
        Converts a FileContextToken to a typed primitive that
        behaves like a primitive, but also contains its location in the file,
        as well the keyword it pertains to and its location in the file.

        :param token: the token to be converted
        :param index: the index of the token
        :param keyword: the keyword it pertains to

        :return: The token as a primitive with context of itself and its keyword
        """
        # pylint: disable=too-many-return-statements, too-many-branches

        if not self._is_in_allowed_values_for_kw(token, index):
            raise ConfigValidationError.from_info(
                ErrorInfo(
                    message=f"{self.kw!r} argument {index!r} must be one of"
                    f" {self.indexed_selection_set[index]!r} was {token.value!r}",
                    filename=token.filename,
                ).set_context(token)
            )

        if not len(self.type_map) > index:
            return StringToken(str(token), token, keyword)
        val_type = self.type_map[index]
        if val_type is None:
            return StringToken(str(token), token, keyword)
        if val_type == SchemaType.CONFIG_BOOL:
            if token.lower() == "true":
                return BoolToken(True, token, keyword)
            elif token.lower() == "false":
                return BoolToken(False, token, keyword)
            else:
                raise ConfigValidationError.from_info(
                    ErrorInfo(
                        message=f"{self.kw!r} must have a boolean value"
                        f" as argument {index + 1!r}",
                        filename=token.filename,
                    ).set_context(token)
                )
        if val_type == SchemaType.CONFIG_INT:
            try:
                return IntToken(int(token), token, keyword)
            except ValueError:
                raise ConfigValidationError.from_info(
                    ErrorInfo(
                        message=f"{self.kw!r} must have an integer value"
                        f" as argument {index + 1!r}",
                        filename=token.filename,
                    ).set_context(token)
                )
        if val_type == SchemaType.CONFIG_FLOAT:
            try:
                return FloatToken(float(token), token, keyword)
            except ValueError:
                raise ConfigValidationError.from_info(
                    ErrorInfo(
                        message=f"{self.kw!r} must have a number "
                        f"as argument {index + 1!r}",
                        filename=token.filename,
                    ).set_context(token)
                )
        if val_type in [SchemaType.CONFIG_PATH, SchemaType.CONFIG_EXISTING_PATH]:
            path = str(token)
            if not os.path.isabs(token):
                path = os.path.normpath(
                    os.path.join(os.path.dirname(token.filename), token)
                )
            if val_type == SchemaType.CONFIG_EXISTING_PATH and not os.path.exists(path):
                err = f'Cannot find file or directory "{token.value}" \n'
                if path != token:
                    err += f"The configured value was {path!r} "
                raise ConfigValidationError.from_info(
                    ErrorInfo(message=err, filename=token.filename).set_context(token)
                )
            return StringToken(path, token, keyword)
        if val_type == SchemaType.CONFIG_EXECUTABLE:
            path = str(token)
            if not os.path.isabs(token) and not os.path.exists(token):
                path = shutil.which(token)

            if path is None:
                raise ConfigValidationError.from_info(
                    ErrorInfo(
                        message=f"Could not find executable {token.value!r}",
                        filename=token.filename,
                    ).set_context(token)
                )

            if not os.access(path, os.X_OK):
                context = (
                    f"{token.value!r} which was resolved to {path!r}"
                    if token.value != path
                    else f"{token.value!r}"
                )
                raise ConfigValidationError.from_info(
                    ErrorInfo(
                        message=f"File not executable: {context}",
                        filename=token.filename,
                    ).set_context(token)
                )
            return StringToken(path, token, keyword)
        return StringToken(str(token), token, keyword)

    def apply_constraints(
        self,
        args: List[Any],
        keyword: FileContextToken,
    ) -> Union[List[Any], Any]:
        errors = []

        args_with_context = []
        for i, x in enumerate(args):
            if isinstance(x, FileContextToken):
                try:
                    value_with_context = self.token_to_value_with_context(x, i, keyword)
                    args_with_context.append(value_with_context)
                except ConfigValidationError as err:
                    errors.append(err)
                    continue
            else:
                args_with_context.append(x)

        args = args_with_context

        if self.argc_min != -1 and len(args) < self.argc_min:
            errors.append(
                ErrorInfo(
                    message=f"{self.kw} must have at least {self.argc_min} arguments",
                    filename=keyword.filename,
                ).set_context(StringToken.from_token(keyword))
            )
        elif self.argc_max != -1 and len(args) > self.argc_max:
            errors.append(
                ErrorInfo(
                    message=f"{self.kw} must have maximum {self.argc_max} arguments",
                    filename=keyword.filename,
                ).set_context(StringToken.from_token(keyword))
            )

        elif self.argc_max == 1 and self.argc_min == 1:
            return args[0]

        if len(errors) > 0:
            raise ConfigValidationError.from_collected(errors)

        return args

    def join_args(self, line: List[Any]) -> List[Any]:
        n = self.join_after
        if 0 < n < len(line):
            joined = FileContextToken.join_tokens(line[n:], " ")
            new_line = line[0:n]
            if len(joined) > 0:
                new_line.append(joined)
            return new_line
        return line


def check_required(
    schema: Mapping[str, SchemaItem],
    config_dict: Mapping[str, Instruction],
    filename: str,
) -> None:
    errors = []

    for constraints in schema.values():
        if constraints.required_set and constraints.kw not in config_dict:
            errors.append(
                ErrorInfo(
                    message=f"{constraints.kw} must be set.",
                    filename=filename,
                )
            )

    if len(errors) > 0:
        raise ConfigValidationError(errors=errors)


def float_keyword(keyword: str) -> SchemaItem:
    return SchemaItem(kw=keyword, type_map=[SchemaType.CONFIG_FLOAT])


def int_keyword(keyword: str) -> SchemaItem:
    return SchemaItem(kw=keyword, type_map=[SchemaType.CONFIG_INT])


def string_keyword(keyword: str) -> SchemaItem:
    return SchemaItem(kw=keyword, type_map=[SchemaType.CONFIG_STRING])


def path_keyword(keyword: str) -> SchemaItem:
    return SchemaItem(kw=keyword, type_map=[SchemaType.CONFIG_PATH])


def existing_path_keyword(keyword: str) -> SchemaItem:
    return SchemaItem(kw=keyword, type_map=[SchemaType.CONFIG_EXISTING_PATH])


def single_arg_keyword(keyword: str) -> SchemaItem:
    return SchemaItem(kw=keyword, argc_max=1, argc_min=1)


def num_realizations_keyword() -> SchemaItem:
    return SchemaItem(
        kw=NUM_REALIZATIONS_KEY,
        required_set=True,
        argc_min=1,
        argc_max=1,
        type_map=[SchemaType.CONFIG_INT],
    )


def run_template_keyword() -> SchemaItem:
    return SchemaItem(
        kw=RUN_TEMPLATE_KEY,
        argc_min=2,
        argc_max=CONFIG_DEFAULT_ARG_MAX,
        type_map=[SchemaType.CONFIG_EXISTING_PATH],
        multi_occurrence=True,
    )


def forward_model_keyword() -> SchemaItem:
    return SchemaItem(
        kw=FORWARD_MODEL_KEY,
        argc_min=0,
        argc_max=CONFIG_DEFAULT_ARG_MAX,
        multi_occurrence=True,
        substitute_from=0,
    )


def simulation_job_keyword() -> SchemaItem:
    return SchemaItem(
        kw=SIMULATION_JOB_KEY,
        argc_min=1,
        argc_max=CONFIG_DEFAULT_ARG_MAX,
        multi_occurrence=True,
    )


def data_kw_keyword() -> SchemaItem:
    return SchemaItem(
        kw=DATA_KW_KEY,
        required_set=False,
        argc_min=2,
        argc_max=2,
        multi_occurrence=True,
        substitute_from=2,
    )


def define_keyword() -> SchemaItem:
    return SchemaItem(
        kw=DEFINE_KEY,
        required_set=False,
        argc_min=2,
        argc_max=2,
        multi_occurrence=True,
        substitute_from=2,
        join_after=1,
    )


def history_source_keyword() -> SchemaItem:
    return SchemaItem(
        kw=HISTORY_SOURCE_KEY,
        argc_max=1,
        argc_min=1,
        common_selection_set=["REFCASE_SIMULATED", "REFCASE_HISTORY"],
        required_children_value={
            "REFCASE_SIMULATED": [REFCASE_KEY],
            "REFCASE_HISTORY": [REFCASE_KEY],
        },
    )


def stop_long_running_keyword() -> SchemaItem:
    return SchemaItem(
        kw=STOP_LONG_RUNNING_KEY,
        type_map=[SchemaType.CONFIG_BOOL],
        required_children_value={"TRUE": [MIN_REALIZATIONS_KEY]},
    )


def analysis_copy_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ANALYSIS_COPY_KEY, argc_min=2, argc_max=2, multi_occurrence=True
    )


def update_setting_keyword() -> SchemaItem:
    return SchemaItem(kw=UPDATE_SETTING_KEY, argc_min=2, argc_max=2)


def analysis_set_var_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ANALYSIS_SET_VAR_KEY,
        argc_min=3,
        argc_max=CONFIG_DEFAULT_ARG_MAX,
        multi_occurrence=True,
    )


def hook_workflow_keyword() -> SchemaItem:
    return SchemaItem(
        kw=HOOK_WORKFLOW_KEY,
        argc_min=2,
        argc_max=2,
        type_map=[SchemaType.CONFIG_STRING, SchemaType.CONFIG_STRING],
        indexed_selection_set={
            1: [
                RUN_MODE_PRE_SIMULATION_NAME,
                RUN_MODE_POST_SIMULATION_NAME,
                RUN_MODE_PRE_UPDATE_NAME,
                RUN_MODE_PRE_FIRST_UPDATE_NAME,
                RUN_MODE_POST_UPDATE_NAME,
            ]
        },
        multi_occurrence=True,
    )


def set_env_keyword() -> SchemaItem:
    # You can set environment variables which will be applied to the run-time
    # environment. Can unfortunately not use constructions like
    # PATH=$PATH:/some/new/path, use the UPDATE_PATH function instead.
    return SchemaItem(
        kw=SETENV_KEY,
        argc_min=2,
        argc_max=2,
        expand_envvar=False,
        multi_occurrence=True,
    )


def update_path_keyword() -> SchemaItem:
    # UPDATE_PATH   LD_LIBRARY_PATH   /path/to/some/funky/lib
    # Will prepend "/path/to/some/funky/lib" at the front of LD_LIBRARY_PATH.
    return SchemaItem(
        kw=UPDATE_PATH_KEY,
        argc_min=2,
        argc_max=2,
        expand_envvar=False,
        multi_occurrence=True,
    )


def install_job_keyword() -> SchemaItem:
    return SchemaItem(
        kw=INSTALL_JOB_KEY,
        argc_min=2,
        argc_max=2,
        multi_occurrence=True,
        type_map=[None, SchemaType.CONFIG_EXISTING_PATH],
    )


def load_workflow_keyword() -> SchemaItem:
    return SchemaItem(
        kw=LOAD_WORKFLOW_KEY,
        argc_min=1,
        argc_max=2,
        multi_occurrence=True,
        type_map=[SchemaType.CONFIG_EXISTING_PATH],
    )


def load_workflow_job_keyword() -> SchemaItem:
    return SchemaItem(
        kw=LOAD_WORKFLOW_JOB_KEY,
        argc_min=1,
        argc_max=2,
        multi_occurrence=True,
        type_map=[SchemaType.CONFIG_EXISTING_PATH],
    )


def queue_system_keyword(required: bool) -> SchemaItem:
    return SchemaItem(
        kw=QUEUE_SYSTEM_KEY, required_set=required, argc_min=1, argc_max=1
    )


def queue_option_keyword() -> SchemaItem:
    return SchemaItem(
        kw=QUEUE_OPTION_KEY,
        argc_min=2,
        argc_max=CONFIG_DEFAULT_ARG_MAX,
        indexed_selection_set={0: ["LSF", "LOCAL", "TORQUE", "SLURM"]},
        join_after=2,
        multi_occurrence=True,
    )


def job_script_keyword() -> SchemaItem:
    return SchemaItem(
        kw=JOB_SCRIPT_KEY,
        argc_max=1,
        argc_min=1,
        type_map=[SchemaType.CONFIG_EXECUTABLE],
    )


def gen_kw_keyword() -> SchemaItem:
    return SchemaItem(
        kw=GEN_KW_KEY,
        argc_min=4,
        argc_max=6,
        type_map=[
            None,
            SchemaType.CONFIG_EXISTING_PATH,
            SchemaType.CONFIG_STRING,
            SchemaType.CONFIG_EXISTING_PATH,
        ],
        multi_occurrence=True,
    )


def schedule_prediction_file_keyword() -> SchemaItem:
    return SchemaItem(
        kw=SCHEDULE_PREDICTION_FILE_KEY,
        required_set=False,
        argc_min=1,
        argc_max=3,
        type_map=[SchemaType.CONFIG_EXISTING_PATH],
        deprecated=True,
        deprecate_msg="The SCHEDULE_PREDICTION_FILE config KEY = is deprecated.",
    )


def summary_keyword() -> SchemaItem:
    # can have several summary keys on each line.
    return SchemaItem(
        kw=SUMMARY_KEY,
        required_set=False,
        argc_min=1,
        argc_max=CONFIG_DEFAULT_ARG_MAX,
        multi_occurrence=True,
    )


def surface_keyword() -> SchemaItem:
    return SchemaItem(
        kw=SURFACE_KEY,
        required_set=False,
        argc_min=4,
        argc_max=5,
        multi_occurrence=True,
    )


def field_keyword() -> SchemaItem:
    # the way config info is entered for fields is unfortunate because
    # it is difficult/impossible to let the config system handle run
    # time validation of the input.

    return SchemaItem(
        kw=FIELD_KEY,
        argc_min=2,
        argc_max=CONFIG_DEFAULT_ARG_MAX,
        required_children=[GRID_KEY],
        multi_occurrence=True,
    )


def gen_data_keyword() -> SchemaItem:
    return SchemaItem(
        kw=GEN_DATA_KEY,
        argc_min=1,
        argc_max=CONFIG_DEFAULT_ARG_MAX,
        multi_occurrence=True,
    )


def workflow_job_directory_keyword() -> SchemaItem:
    return SchemaItem(
        kw=WORKFLOW_JOB_DIRECTORY_KEY,
        type_map=[SchemaType.CONFIG_PATH],
        multi_occurrence=True,
    )


def install_job_directory_keyword() -> SchemaItem:
    return SchemaItem(
        kw=INSTALL_JOB_DIRECTORY_KEY,
        type_map=[SchemaType.CONFIG_PATH],
        multi_occurrence=True,
    )


def init_site_config() -> Dict[str, SchemaItem]:
    schema = {}
    for item in [
        int_keyword(MAX_SUBMIT_KEY),
        int_keyword(NUM_CPU_KEY),
        queue_system_keyword(True),
        queue_option_keyword(),
        job_script_keyword(),
        workflow_job_directory_keyword(),
        load_workflow_keyword(),
        load_workflow_job_keyword(),
        set_env_keyword(),
        update_path_keyword(),
        install_job_keyword(),
        install_job_directory_keyword(),
        hook_workflow_keyword(),
    ]:
        schema[item.kw] = item
        if item.kw in ALIASES:
            for name in ALIASES[item.kw]:
                schema[name] = item
    return schema


def init_user_config() -> Dict[str, SchemaItem]:
    schema = {}
    for item in [
        workflow_job_directory_keyword(),
        load_workflow_keyword(),
        load_workflow_job_keyword(),
        float_keyword(ENKF_ALPHA_KEY),
        float_keyword(STD_CUTOFF_KEY),
        update_setting_keyword(),
        string_keyword(UPDATE_LOG_PATH_KEY),
        string_keyword(MIN_REALIZATIONS_KEY),
        int_keyword(MAX_RUNTIME_KEY),
        string_keyword(ANALYSIS_SELECT_KEY),
        stop_long_running_keyword(),
        analysis_copy_keyword(),
        analysis_set_var_keyword(),
        string_keyword(ITER_CASE_KEY),
        int_keyword(ITER_COUNT_KEY),
        int_keyword(ITER_RETRY_COUNT_KEY),
        # the two fault types are just added to the config object only to
        # be able to print suitable messages before exiting.
        gen_kw_keyword(),
        schedule_prediction_file_keyword(),
        string_keyword(GEN_KW_TAG_FORMAT_KEY),
        gen_data_keyword(),
        summary_keyword(),
        surface_keyword(),
        field_keyword(),
        single_arg_keyword(ECLBASE_KEY),
        existing_path_keyword(DATA_FILE_KEY),
        existing_path_keyword(GRID_KEY),
        path_keyword(REFCASE_KEY),
        string_keyword(RANDOM_SEED_KEY),
        num_realizations_keyword(),
        run_template_keyword(),
        path_keyword(RUNPATH_KEY),
        path_keyword(DATA_ROOT_KEY),
        path_keyword(ENSPATH_KEY),
        single_arg_keyword(JOBNAME_KEY),
        forward_model_keyword(),
        simulation_job_keyword(),
        data_kw_keyword(),
        define_keyword(),
        existing_path_keyword(OBS_CONFIG_KEY),
        existing_path_keyword(TIME_MAP_KEY),
        single_arg_keyword(GEN_KW_EXPORT_NAME_KEY),
        history_source_keyword(),
        path_keyword(RUNPATH_FILE_KEY),
        int_keyword(MAX_SUBMIT_KEY),
        int_keyword(NUM_CPU_KEY),
        queue_system_keyword(False),
        queue_option_keyword(),
        job_script_keyword(),
        load_workflow_job_keyword(),
        set_env_keyword(),
        update_path_keyword(),
        path_keyword(LICENSE_PATH_KEY),
        install_job_keyword(),
        install_job_directory_keyword(),
        hook_workflow_keyword(),
        existing_path_keyword(CONFIG_DIRECTORY_KEY),
    ]:
        schema[item.kw] = item
        if item.kw in ALIASES:
            for name in ALIASES[item.kw]:
                schema[name] = item
    return schema
