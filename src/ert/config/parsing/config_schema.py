from .config_dict import ConfigDict
from .config_keywords import ConfigKeys
from .config_schema_deprecations import deprecated_keywords_list
from .config_schema_item import (
    SchemaItem,
    Varies,
    existing_path_inline_keyword,
    existing_path_keyword,
    float_keyword,
    int_keyword,
    path_keyword,
    positive_float_keyword,
    positive_int_keyword,
    single_arg_keyword,
    string_keyword,
)
from .history_source import HistorySource
from .hook_runtime import HookRuntime
from .queue_system import QueueSystem, QueueSystemWithGeneric
from .schema_dict import SchemaItemDict
from .schema_item_type import SchemaItemType

ConfigAliases = {ConfigKeys.NUM_REALIZATIONS: ["NUM_REALISATIONS"]}


def num_realizations_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.NUM_REALIZATIONS,
        required_set=True,
        type_map=[SchemaItemType.INT],
    )


def run_template_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.RUN_TEMPLATE,
        argc_min=2,
        argc_max=2,
        type_map=[SchemaItemType.EXISTING_PATH],
        multi_occurrence=True,
    )


def forward_model_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.FORWARD_MODEL,
        argc_min=0,
        argc_max=None,
        multi_occurrence=True,
        substitute_from=2,
    )


def data_kw_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.DATA_KW,
        required_set=False,
        argc_min=2,
        argc_max=2,
        multi_occurrence=True,
        substitute_from=2,
    )


def define_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.DEFINE,
        required_set=False,
        argc_min=2,
        argc_max=2,
        multi_occurrence=True,
        substitute_from=2,
        join_after=1,
    )


def history_source_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.HISTORY_SOURCE,
        argc_min=1,
        argc_max=1,
        type_map=[HistorySource],
        required_children_value={
            "REFCASE_SIMULATED": [ConfigKeys.REFCASE],
            "REFCASE_HISTORY": [ConfigKeys.REFCASE],
        },
    )


def stop_long_running_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.STOP_LONG_RUNNING,
        type_map=[SchemaItemType.BOOL],
        required_children_value={"TRUE": [ConfigKeys.MIN_REALIZATIONS]},
    )


def analysis_set_var_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.ANALYSIS_SET_VAR,
        argc_min=3,
        argc_max=None,
        multi_occurrence=True,
    )


def hook_workflow_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.HOOK_WORKFLOW,
        argc_min=2,
        argc_max=2,
        type_map=[SchemaItemType.STRING, HookRuntime],
        multi_occurrence=True,
    )


def set_env_keyword() -> SchemaItem:
    # You can set environment variables which will be applied to the run-time
    # environment.
    return SchemaItem(
        kw=ConfigKeys.SETENV,
        argc_min=2,
        argc_max=2,
        expand_envvar=False,
        multi_occurrence=True,
    )


def install_job_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.INSTALL_JOB,
        argc_min=2,
        argc_max=2,
        multi_occurrence=True,
        type_map=[None, SchemaItemType.EXISTING_PATH_INLINE],
    )


def load_workflow_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.LOAD_WORKFLOW,
        argc_max=2,
        multi_occurrence=True,
        type_map=[SchemaItemType.EXISTING_PATH],
    )


def load_workflow_job_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.LOAD_WORKFLOW_JOB,
        argc_max=2,
        multi_occurrence=True,
        type_map=[SchemaItemType.EXISTING_PATH],
    )


def queue_system_keyword(required: bool) -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.QUEUE_SYSTEM,
        argc_min=1,
        argc_max=1,
        type_map=[QueueSystem],
        required_set=required,
    )


def queue_option_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.QUEUE_OPTION,
        argc_min=2,
        argc_max=None,
        join_after=2,
        type_map=[QueueSystemWithGeneric, SchemaItemType.STRING, SchemaItemType.STRING],
        multi_occurrence=True,
    )


def job_script_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.JOB_SCRIPT,
        type_map=[SchemaItemType.EXECUTABLE],
    )


def gen_kw_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.GEN_KW,
        argc_min=2,
        argc_max=6,
        options_after=Varies(4),
        type_map=[
            None,
            SchemaItemType.EXISTING_PATH_INLINE,
            SchemaItemType.STRING,
            SchemaItemType.EXISTING_PATH_INLINE,
        ],
        multi_occurrence=True,
    )


def summary_keyword() -> SchemaItem:
    # can have several summary keys on each line.
    return SchemaItem(
        kw=ConfigKeys.SUMMARY,
        required_set=False,
        argc_max=None,
        required_children=[ConfigKeys.ECLBASE],
        multi_occurrence=True,
    )


def surface_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.SURFACE,
        required_set=False,
        argc_min=2,
        argc_max=2,
        options_after=1,
        multi_occurrence=True,
    )


def field_keyword() -> SchemaItem:
    # the way config info is entered for fields is unfortunate because
    # it is difficult/impossible to let the config system handle run
    # time validation of the input.

    return SchemaItem(
        kw=ConfigKeys.FIELD,
        argc_min=3,
        argc_max=None,
        options_after=3,
        required_children=[ConfigKeys.GRID],
        multi_occurrence=True,
    )


def gen_data_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.GEN_DATA,
        argc_max=None,
        options_after=1,
        multi_occurrence=True,
    )


def workflow_job_directory_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.WORKFLOW_JOB_DIRECTORY,
        type_map=[SchemaItemType.PATH],
        multi_occurrence=True,
    )


def install_job_directory_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.INSTALL_JOB_DIRECTORY,
        type_map=[SchemaItemType.PATH],
        multi_occurrence=True,
    )


def design_matrix_keyword() -> SchemaItem:
    return SchemaItem(
        kw=ConfigKeys.DESIGN_MATRIX,
        argc_min=1,
        argc_max=3,
        type_map=[
            SchemaItemType.EXISTING_PATH,
            SchemaItemType.STRING,
            SchemaItemType.STRING,
        ],
        options_after=1,
        multi_occurrence=True,
    )


class ConfigSchemaDict(SchemaItemDict):
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


def init_site_config_schema() -> ConfigSchemaDict:
    schema = ConfigSchemaDict()
    for item in [
        positive_int_keyword(ConfigKeys.MAX_SUBMIT),
        positive_int_keyword(ConfigKeys.NUM_CPU),
        string_keyword(ConfigKeys.REALIZATION_MEMORY),
        queue_system_keyword(False),
        queue_option_keyword(),
        job_script_keyword(),
        workflow_job_directory_keyword(),
        load_workflow_keyword(),
        load_workflow_job_keyword(),
        set_env_keyword(),
        install_job_keyword(),
        install_job_directory_keyword(),
        hook_workflow_keyword(),
    ]:
        schema[item.kw] = item
        if item.kw in ConfigAliases:
            for name in ConfigAliases[ConfigKeys(item.kw)]:
                schema[name] = item

    return schema


def init_user_config_schema() -> ConfigSchemaDict:
    schema = ConfigSchemaDict()
    for item in [
        workflow_job_directory_keyword(),
        load_workflow_keyword(),
        load_workflow_job_keyword(),
        float_keyword(ConfigKeys.ENKF_ALPHA),
        positive_float_keyword(ConfigKeys.STD_CUTOFF),
        float_keyword(ConfigKeys.SUBMIT_SLEEP),
        string_keyword(keyword=ConfigKeys.UPDATE_LOG_PATH),
        string_keyword(ConfigKeys.MIN_REALIZATIONS),
        int_keyword(ConfigKeys.MAX_RUNTIME),
        stop_long_running_keyword(),
        analysis_set_var_keyword(),
        # the two fault types are just added to the config object only to
        # be able to print suitable messages before exiting.
        gen_kw_keyword(),
        gen_data_keyword(),
        summary_keyword(),
        surface_keyword(),
        field_keyword(),
        single_arg_keyword(ConfigKeys.ECLBASE),
        existing_path_keyword(ConfigKeys.DATA_FILE),
        existing_path_keyword(ConfigKeys.GRID),
        path_keyword(ConfigKeys.REFCASE),
        int_keyword(ConfigKeys.RANDOM_SEED),
        num_realizations_keyword(),
        run_template_keyword(),
        path_keyword(ConfigKeys.RUNPATH),
        path_keyword(ConfigKeys.ENSPATH),
        single_arg_keyword(ConfigKeys.JOBNAME),
        forward_model_keyword(),
        data_kw_keyword(),
        define_keyword(),
        existing_path_inline_keyword(ConfigKeys.OBS_CONFIG),
        existing_path_inline_keyword(ConfigKeys.TIME_MAP),
        single_arg_keyword(ConfigKeys.GEN_KW_EXPORT_NAME),
        history_source_keyword(),
        path_keyword(ConfigKeys.RUNPATH_FILE),
        positive_int_keyword(ConfigKeys.MAX_SUBMIT),
        positive_int_keyword(ConfigKeys.NUM_CPU),
        positive_int_keyword(ConfigKeys.MAX_RUNNING),
        string_keyword(ConfigKeys.REALIZATION_MEMORY),
        design_matrix_keyword(),
        queue_system_keyword(False),
        queue_option_keyword(),
        job_script_keyword(),
        load_workflow_job_keyword(),
        set_env_keyword(),
        install_job_keyword(),
        install_job_directory_keyword(),
        hook_workflow_keyword(),
    ]:
        schema[item.kw] = item
        if item.kw in ConfigAliases:
            for name in ConfigAliases[ConfigKeys(item.kw)]:
                schema[name] = item

        schema.add_deprecations(deprecated_keywords_list)

    return schema
