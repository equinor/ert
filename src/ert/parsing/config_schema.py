from typing import Dict


from .config_keywords import (
    ConfigKeys,
    RunModes,
    QueueOptions,
)
from .schema_dict import SchemaItemDict
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

CONFIG_DEFAULT_ARG_MAX = -1
CONFIG_DEFAULT_ARG_MIN = -1
ConfigAliases = {ConfigKeys.NUM_REALIZATIONS_KEY: ["NUM_REALISATIONS"]}


def num_realizations_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.NUM_REALIZATIONS_KEY,
        required_set=True,
        argc_min=1,
        argc_max=1,
        type_map=[SchemaItemType.INT],
    )


def run_template_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.RUN_TEMPLATE_KEY,
        argc_min=2,
        argc_max=CONFIG_DEFAULT_ARG_MAX,
        type_map=[SchemaItemType.EXISTING_PATH],
        multi_occurrence=True,
    )


def forward_model_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.FORWARD_MODEL_KEY,
        argc_min=0,
        argc_max=CONFIG_DEFAULT_ARG_MAX,
        multi_occurrence=True,
        substitute_from=0,
    )


def simulation_job_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.SIMULATION_JOB_KEY,
        argc_min=1,
        argc_max=CONFIG_DEFAULT_ARG_MAX,
        multi_occurrence=True,
    )


def data_kw_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.DATA_KW_KEY,
        required_set=False,
        argc_min=2,
        argc_max=2,
        multi_occurrence=True,
        substitute_from=2,
    )


def define_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.DEFINE_KEY,
        required_set=False,
        argc_min=2,
        argc_max=2,
        multi_occurrence=True,
        substitute_from=2,
        join_after=1,
    )


def history_source_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.HISTORY_SOURCE_KEY,
        argc_max=1,
        argc_min=1,
        common_selection_set=["REFCASE_SIMULATED", "REFCASE_HISTORY"],
        required_children_value={
            "REFCASE_SIMULATED": [ConfigKeys.REFCASE_KEY],
            "REFCASE_HISTORY": [ConfigKeys.REFCASE_KEY],
        },
    )


def stop_long_running_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.STOP_LONG_RUNNING_KEY,
        type_map=[SchemaItemType.BOOL],
        required_children_value={"TRUE": [ConfigKeys.MIN_REALIZATIONS_KEY]},
    )


def analysis_copy_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.ANALYSIS_COPY_KEY, argc_min=2, argc_max=2, multi_occurrence=True
    )


def update_setting_keyword() -> SchemaItem:
    return SchemaItem(kw=ConfigKeys.UPDATE_SETTING_KEY, argc_min=2, argc_max=2)


def analysis_set_var_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.ANALYSIS_SET_VAR_KEY,
        argc_min=3,
        argc_max=CONFIG_DEFAULT_ARG_MAX,
        multi_occurrence=True,
    )


def hook_workflow_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.HOOK_WORKFLOW_KEY,
        argc_min=2,
        argc_max=2,
        type_map=[SchemaItemType.STRING, SchemaItemType.STRING],
        indexed_selection_set={1: list(RunModes)},
        multi_occurrence=True,
    )


def set_env_keyword() -> SchemaItem:
    # You can set environment variables which will be applied to the run-time
    # environment. Can unfortunately not use constructions like
    # PATH=$PATH:/some/new/path, use the UPDATE_PATH function instead.
    return SchemaItem(
        kw=ConfigKeys.SETENV_KEY,
        argc_min=2,
        argc_max=2,
        expand_envvar=False,
        multi_occurrence=True,
    )


def update_path_keyword() -> SchemaItem:
    # UPDATE_PATH   LD_LIBRARY_PATH   /path/to/some/funky/lib
    # Will prepend "/path/to/some/funky/lib" at the front of LD_LIBRARY_PATH.
    return SchemaItem(
        kw=ConfigKeys.UPDATE_PATH_KEY,
        argc_min=2,
        argc_max=2,
        expand_envvar=False,
        multi_occurrence=True,
    )


def install_job_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.INSTALL_JOB_KEY,
        argc_min=2,
        argc_max=2,
        multi_occurrence=True,
        type_map=[None, SchemaItemType.EXISTING_PATH],
    )


def load_workflow_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.LOAD_WORKFLOW_KEY,
        argc_min=1,
        argc_max=2,
        multi_occurrence=True,
        type_map=[SchemaItemType.EXISTING_PATH],
    )


def load_workflow_job_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.LOAD_WORKFLOW_JOB_KEY,
        argc_min=1,
        argc_max=2,
        multi_occurrence=True,
        type_map=[SchemaItemType.EXISTING_PATH],
    )


def queue_system_keyword(required: bool) -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.QUEUE_SYSTEM, required_set=required, argc_min=1, argc_max=1
    )


def queue_option_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.QUEUE_OPTION_KEY,
        argc_min=2,
        argc_max=CONFIG_DEFAULT_ARG_MAX,
        indexed_selection_set={0: list(QueueOptions)},
        join_after=2,
        multi_occurrence=True,
    )


def job_script_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.JOB_SCRIPT_KEY,
        argc_max=1,
        argc_min=1,
        type_map=[SchemaItemType.EXECUTABLE],
    )


def gen_kw_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.GEN_KW_KEY,
        argc_min=4,
        argc_max=6,
        type_map=[
            None,
            SchemaItemType.EXISTING_PATH,
            SchemaItemType.STRING,
            SchemaItemType.EXISTING_PATH,
        ],
        multi_occurrence=True,
    )


def schedule_prediction_file_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.SCHEDULE_PREDICTION_FILE_KEY,
        required_set=False,
        argc_min=1,
        argc_max=3,
        type_map=[SchemaItemType.STRING],
        deprecated=True,
        deprecate_msg="The SCHEDULE_PREDICTION_FILE config KEY has been removed.",
    )


def summary_keyword() -> SchemaItem:
    # can have several summary keys on each line.
    return SchemaItem(
        kw=ConfigKeys.SUMMARY_KEY,
        required_set=False,
        argc_min=1,
        argc_max=CONFIG_DEFAULT_ARG_MAX,
        multi_occurrence=True,
    )


def surface_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.SURFACE_KEY,
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
        kw=ConfigKeys.FIELD_KEY,
        argc_min=2,
        argc_max=CONFIG_DEFAULT_ARG_MAX,
        required_children=[ConfigKeys.GRID_KEY],
        multi_occurrence=True,
    )


def gen_data_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.GEN_DATA_KEY,
        argc_min=1,
        argc_max=CONFIG_DEFAULT_ARG_MAX,
        multi_occurrence=True,
    )


def workflow_job_directory_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.WORKFLOW_JOB_DIRECTORY_KEY,
        type_map=[SchemaItemType.PATH],
        multi_occurrence=True,
    )


def install_job_directory_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.INSTALL_JOB_DIRECTORY_KEY,
        type_map=[SchemaItemType.PATH],
        multi_occurrence=True,
    )


class ConfigSchemaDict(SchemaItemDict):
    pass


def init_site_config() -> SchemaItemDict:
    schema = SchemaItemDict()
    for item in [
        int_keyword(ConfigKeys.MAX_SUBMIT_KEY),
        int_keyword(ConfigKeys.NUM_CPU_KEY),
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
        if item.kw in ConfigAliases:
            for name in ConfigAliases[item.kw]:
                schema[name] = item

    return schema


def init_user_config() -> SchemaItemDict:
    schema = SchemaItemDict()
    for item in [
        workflow_job_directory_keyword(),
        load_workflow_keyword(),
        load_workflow_job_keyword(),
        float_keyword(ConfigKeys.ENKF_ALPHA_KEY),
        float_keyword(ConfigKeys.STD_CUTOFF_KEY),
        update_setting_keyword(),
        string_keyword(keyword=ConfigKeys.UPDATE_LOG_PATH_KEY),
        string_keyword(ConfigKeys.MIN_REALIZATIONS_KEY),
        int_keyword(ConfigKeys.MAX_RUNTIME_KEY),
        string_keyword(ConfigKeys.ANALYSIS_SELECT_KEY),
        stop_long_running_keyword(),
        analysis_copy_keyword(),
        analysis_set_var_keyword(),
        string_keyword(ConfigKeys.ITER_CASE_KEY),
        int_keyword(ConfigKeys.ITER_COUNT_KEY),
        int_keyword(ConfigKeys.ITER_RETRY_COUNT_KEY),
        # the two fault types are just added to the config object only to
        # be able to print suitable messages before exiting.
        gen_kw_keyword(),
        schedule_prediction_file_keyword(),
        string_keyword(ConfigKeys.GEN_KW_TAG_FORMAT_KEY),
        gen_data_keyword(),
        summary_keyword(),
        surface_keyword(),
        field_keyword(),
        single_arg_keyword(ConfigKeys.ECLBASE_KEY),
        existing_path_keyword(ConfigKeys.DATA_FILE_KEY),
        existing_path_keyword(ConfigKeys.GRID_KEY),
        path_keyword(ConfigKeys.REFCASE_KEY),
        string_keyword(ConfigKeys.RANDOM_SEED_KEY),
        num_realizations_keyword(),
        run_template_keyword(),
        path_keyword(ConfigKeys.RUNPATH_KEY),
        path_keyword(ConfigKeys.DATA_ROOT_KEY),
        path_keyword(ConfigKeys.ENSPATH_KEY),
        single_arg_keyword(ConfigKeys.JOBNAME_KEY),
        forward_model_keyword(),
        simulation_job_keyword(),
        data_kw_keyword(),
        define_keyword(),
        existing_path_keyword(ConfigKeys.OBS_CONFIG_KEY),
        existing_path_keyword(ConfigKeys.TIME_MAP_KEY),
        single_arg_keyword(ConfigKeys.GEN_KW_EXPORT_NAME_KEY),
        history_source_keyword(),
        path_keyword(ConfigKeys.RUNPATH_FILE_KEY),
        int_keyword(ConfigKeys.MAX_SUBMIT_KEY),
        int_keyword(ConfigKeys.NUM_CPU_KEY),
        queue_system_keyword(False),
        queue_option_keyword(),
        job_script_keyword(),
        load_workflow_job_keyword(),
        set_env_keyword(),
        update_path_keyword(),
        path_keyword(ConfigKeys.LICENSE_PATH_KEY),
        install_job_keyword(),
        install_job_directory_keyword(),
        hook_workflow_keyword(),
        existing_path_keyword(ConfigKeys.CONFIG_DIRECTORY_KEY),
    ]:
        schema[item.kw] = item
        if item.kw in ConfigAliases:
            for name in ConfigAliases[item.kw]:
                schema[name] = item

    return schema
