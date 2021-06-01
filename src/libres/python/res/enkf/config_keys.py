#  Copyright (C) 2017  Equinor ASA, Norway.
#
#  The file 'config_keys.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.

from res import ResPrototype


class ConfigKeys:

    _config_directory_key = ResPrototype(
        "char* config_keys_get_config_directory_key()", bind=False
    )
    _config_file_key = ResPrototype(
        "char* config_keys_get_config_file_key()", bind=False
    )
    _queue_system_key = ResPrototype(
        "char* config_keys_get_queue_system_key()", bind=False
    )
    _run_template_key = ResPrototype(
        "char* config_keys_get_run_template_key()", bind=False
    )
    _gen_kw_key = ResPrototype("char* config_keys_get_gen_kw_key()", bind=False)
    _history_source_key = ResPrototype(
        "char* config_keys_get_history_source_key()", bind=False
    )
    _queue_option_key = ResPrototype(
        "char* config_keys_get_queue_option_key()", bind=False
    )
    _install_job_key = ResPrototype(
        "char* config_keys_get_install_job_key()", bind=False
    )
    _path_key = ResPrototype("char* config_keys_get_path_key()", bind=False)
    _show_refcase_key = ResPrototype(
        "char* config_keys_get_show_refcase_key()", bind=False
    )
    _show_history_key = ResPrototype(
        "char* config_keys_get_show_history_key()", bind=False
    )
    _install_job_directory_key = ResPrototype(
        "char* config_keys_get_install_job_directory_key()", bind=False
    )

    _log_file_key = ResPrototype("char* config_keys_get_log_file_key()", bind=False)
    _log_level_key = ResPrototype("char* config_keys_get_log_level_key()", bind=False)
    _update_log_path = ResPrototype(
        "char* config_keys_get_update_log_path_key()", bind=False
    )
    _summary = ResPrototype("char* config_keys_get_summary_key()", bind=False)

    _max_runtime = ResPrototype("char* config_keys_get_max_runtime_key()", bind=False)
    _min_realizations = ResPrototype(
        "char* config_keys_get_min_realizations_key()", bind=False
    )
    _max_submit = ResPrototype("char* config_keys_get_max_submit_key()", bind=False)
    _umask = ResPrototype("char* config_keys_get_umask_key()", bind=False)
    _data_kw_key = ResPrototype("char* config_keys_get_data_kw_key()", bind=False)
    _runpath_file = ResPrototype("char* config_keys_get_runpath_file_key()", bind=False)

    # ************* ECL config  *************
    _eclbase = ResPrototype("char* config_keys_get_eclbase_key()", bind=False)
    _data_file = ResPrototype("char* config_keys_get_data_file_key()", bind=False)
    _grid = ResPrototype("char* config_keys_get_grid_key()", bind=False)
    _refcase = ResPrototype("char* config_keys_get_refcase_key()", bind=False)
    _refcase_list = ResPrototype("char* config_keys_get_refcase_list_key()", bind=False)
    _end_date = ResPrototype("char* config_keys_get_end_date_key()", bind=False)
    _schedule_prediction_file = ResPrototype(
        "char* config_keys_get_schedule_prediction_file_key()", bind=False
    )
    # ************* ECL config  *************

    # ************* Model config  *************
    _num_realizations = ResPrototype(
        "char* config_keys_get_num_realizations_key()", bind=False
    )
    _enspath = ResPrototype("char* config_keys_get_enspath_key()", bind=False)
    _history_source = ResPrototype(
        "char* config_keys_get_history_source_key()", bind=False
    )
    _obs_config = ResPrototype("char* config_keys_get_obs_config_key()", bind=False)
    _time_map = ResPrototype("char* config_keys_get_time_map_key()", bind=False)
    _jobname = ResPrototype("char* config_keys_get_jobname_key()", bind=False)
    _forward_model_key = ResPrototype(
        "char* config_keys_get_forward_model_key()", bind=False
    )
    _simulation_job_key = ResPrototype(
        "char* config_keys_get_simulation_job_key()", bind=False
    )
    _max_resample_key = ResPrototype(
        "char* config_keys_get_max_resample_key()", bind=False
    )
    _data_root_key = ResPrototype("char* config_keys_get_data_root_key()", bind=False)
    _rftpath_key = ResPrototype("char* config_keys_get_rftpath_key()", bind=False)
    _gen_kw_export_name_key = ResPrototype(
        "char* config_keys_get_gen_kw_export_name_key()", bind=False
    )
    _runpath = ResPrototype("char* config_keys_get_runpath_key()", bind=False)
    # ************* Model config  *************

    _gen_data = ResPrototype("char* config_keys_get_gen_data_key()", bind=False)
    _result_file = ResPrototype("char* config_keys_get_result_file()", bind=False)
    _report_steps = ResPrototype("char* config_keys_get_report_steps()", bind=False)
    _input_format = ResPrototype("char* config_keys_get_input_format()", bind=False)
    _ecl_file = ResPrototype("char* config_keys_get_ecl_file()", bind=False)
    _output_format = ResPrototype("char* config_keys_get_output_format()", bind=False)
    _init_files = ResPrototype("char* config_keys_get_init_files()", bind=False)
    _random_seed = ResPrototype("char* config_keys_get_random_seed()", bind=False)
    _license_path_key = ResPrototype(
        "char* config_keys_get_license_path_key()", bind=False
    )
    _setenv_key = ResPrototype("char* config_keys_get_setenv_key()", bind=False)
    _job_script_key = ResPrototype("char* config_keys_get_job_script_key()", bind=False)
    _num_cpu_key = ResPrototype("char* config_keys_get_num_cpu_key()", bind=False)
    _define_key = ResPrototype("char* config_keys_get_define_key()", bind=False)
    _load_workflow_job_key = ResPrototype(
        "char* config_keys_get_load_workflow_job_key()", bind=False
    )
    _workflow_job_directory_key = ResPrototype(
        "char* config_keys_get_workflow_job_directory_key()", bind=False
    )
    _load_workflow_key = ResPrototype(
        "char* config_keys_get_load_workflow_key()", bind=False
    )

    # hook_manager config keys
    _hook_workflow_key = ResPrototype(
        "char* config_keys_get_hook_workflow_key()", bind=False
    )
    HOOK_WORKFLOW_KEY = _hook_workflow_key()
    # hook_manager config keys

    # analysis_iter_config keys
    _iter_case_key = ResPrototype("char* config_keys_get_iter_case_key()", bind=False)
    _iter_count_key = ResPrototype("char* config_keys_get_iter_count_key()", bind=False)
    _iter_retry_count_key = ResPrototype(
        "char* config_keys_get_iter_retry_count_key()", bind=False
    )

    ITER_CASE = _iter_case_key()
    ITER_COUNT = _iter_count_key()
    ITER_RETRY_COUNT = _iter_retry_count_key()
    # analysis_iter_config keys

    # analysis_config keys
    _alpha_key = ResPrototype("char* config_keys_get_alpha()", bind=False)
    _std_cutoff_key = ResPrototype("char* config_keys_get_std_cutoff()", bind=False)
    _stop_long_running = ResPrototype(
        "char* config_keys_get_stop_long_running()", bind=False
    )
    _single_node_update = ResPrototype(
        "char* config_keys_get_single_node_update()", bind=False
    )
    _std_correlated_obs = ResPrototype(
        "char* config_keys_get_std_scale_correlated_obs()", bind=False
    )

    _rerun_key = ResPrototype("char* config_keys_get_rerun()", bind=False)
    _rerun_start_key = ResPrototype("char* config_keys_get_rerun_start()", bind=False)
    _merge_observations = ResPrototype(
        "char* config_keys_get_merge_observations()", bind=False
    )
    _analysis_load = ResPrototype("char* config_keys_get_analysis_load()", bind=False)
    _analysis_copy = ResPrototype("char* config_keys_get_analysis_copy()", bind=False)
    _analysis_select = ResPrototype(
        "char* config_keys_get_analysis_select()", bind=False
    )
    _analysis_set_var = ResPrototype(
        "char* config_keys_get_analysis_set_var()", bind=False
    )

    # slurm options
    _sbatch_option = ResPrototype(
        "char* config_keys_get_slurm_sbatch_option()", bind=False
    )
    _scancel_option = ResPrototype(
        "char* config_keys_get_slurm_scancel_option()", bind=False
    )
    _scontrol_option = ResPrototype(
        "char* config_keys_get_slurm_scontrol_option()", bind=False
    )
    _squeue_option = ResPrototype(
        "char* config_keys_get_slurm_squeue_option()", bind=False
    )
    _partition_option = ResPrototype(
        "char* config_keys_get_slurm_partition_option()", bind=False
    )
    _squeue_timeout_option = ResPrototype(
        "char* config_keys_get_slurm_squeue_timeout_option()", bind=False
    )
    _max_runtime_option = ResPrototype(
        "char* config_keys_get_slurm_max_runtime_option()", bind=False
    )
    _memory_option = ResPrototype(
        "char* config_keys_get_slurm_memory_option()", bind=False
    )
    _memory_per_cpu_option = ResPrototype(
        "char* config_keys_get_slurm_memory_per_cpu_option()", bind=False
    )
    _exclude_host_option = ResPrototype(
        "char* config_keys_get_slurm_exclude_host_option()", bind=False
    )
    _include_host_option = ResPrototype(
        "char* config_keys_get_slurm_include_host_option()", bind=False
    )

    ALPHA_KEY = _alpha_key()
    STD_CUTOFF_KEY = _std_cutoff_key()
    STOP_LONG_RUNNING = _stop_long_running()
    SINGLE_NODE_UPDATE = _single_node_update()
    STD_CORRELATED_OBS = _std_correlated_obs()
    GLOBAL_STD_SCALING = "GLOBAL_STD_SCALING"
    RERUN_KEY = _rerun_key()
    RERUN_START_KEY = _rerun_start_key()
    MERGE_OBSERVATIONS = _merge_observations()
    ANALYSIS_LOAD = _analysis_load()
    USER_NAME = "USER_NAME"
    LIB_NAME = "LIB_NAME"
    ANALYSIS_COPY = _analysis_copy()
    SRC_NAME = "SRC_NAME"
    DST_NAME = "DST_NAME"
    ANALYSIS_SET_VAR = _analysis_set_var()
    MODULE_NAME = "MODULE_NAME"
    VAR_NAME = "VAR_NAME"
    ANALYSIS_SELECT = _analysis_select()
    # analysis_config keys

    ARGLIST = "ARGLIST"
    CONFIG_DIRECTORY = _config_directory_key()
    CONFIG_FILE_KEY = _config_file_key()
    DEFINES = "DEFINES"
    DEFINE_KEY = _define_key()
    INTERNALS = "INTERNALS"
    SIMULATION = "SIMULATION"
    LOGGING = "LOGGING"
    SEED = "SEED"
    QUEUE_SYSTEM = _queue_system_key()
    RUN_TEMPLATE = _run_template_key()
    _template_key = ResPrototype("char* config_keys_get_template_key()", bind=False)
    TEMPLATE = _template_key()
    EXPORT = "EXPORT"
    GEN_KW = _gen_kw_key()
    NAME = "NAME"
    OUT_FILE = "OUT_FILE"
    PARAMETER_FILE = "PARAMETER_FILE"
    PATH = "PATH"
    QUEUE_OPTION = _queue_option_key()
    DRIVER_NAME = "DRIVER_NAME"
    OPTION = "OPTION"
    VALUE = "VALUE"
    INSTALL_JOB = _install_job_key()
    PATH_KEY = _path_key
    SHOW_REFCASE_KEY = _show_refcase_key
    SHOW_HISTORY_KEY = _show_history_key
    LOG_FILE = _log_file_key()
    LOG_LEVEL = _log_level_key()
    UPDATE_LOG_PATH = _update_log_path()
    RANDOM_SEED = _random_seed()
    SUMMARY = _summary()
    MAX_RUNTIME = _max_runtime()
    MIN_REALIZATIONS = _min_realizations()
    MAX_SUBMIT = _max_submit()
    UMASK = _umask()
    MAX_RUNNING = "MAX_RUNNING"
    DATA_KW_KEY = _data_kw_key()
    RUNPATH_FILE = _runpath_file()
    RUNPATH_LIST_FILE = ".ert_runpath_list"
    GEN_DATA = _gen_data()
    RESULT_FILE = _result_file()
    REPORT_STEPS = _report_steps()
    INPUT_FORMAT = _input_format()
    ECL_FILE = _ecl_file()
    OUTPUT_FORMAT = _output_format()
    INIT_FILES = _init_files()
    LICENSE_PATH = _license_path_key()
    INSTALL_JOB_DIRECTORY = _install_job_directory_key()
    SETENV = _setenv_key()
    JOB_SCRIPT = _job_script_key()
    NUM_CPU = _num_cpu_key()
    USER_MODE = "USER_MODE"
    LOAD_WORKFLOW_JOB = _load_workflow_job_key()
    WORKFLOW_JOB_DIRECTORY = _workflow_job_directory_key()
    LOAD_WORKFLOW = _load_workflow_key()

    # ************* ECL config  *************
    ECLBASE = _eclbase()
    DATA_FILE = _data_file()
    GRID = _grid()
    REFCASE = _refcase()
    REFCASE_LIST = _refcase_list()
    END_DATE = _end_date()
    SCHEDULE_PREDICTION_FILE = _schedule_prediction_file()
    # ************* ECL config  *************

    # ************* Model config  *************
    JOBNAME = _jobname()
    FORWARD_MODEL = _forward_model_key()
    SIMULATION_JOB = _simulation_job_key()
    RUNPATH = _runpath()
    MAX_RESAMPLE = _max_resample_key()
    DATAROOT = _data_root_key()
    RFTPATH = _rftpath_key()
    GEN_KW_EXPORT_NAME = _gen_kw_export_name_key()
    NUM_REALIZATIONS = _num_realizations()
    ENSPATH = _enspath()
    HISTORY_SOURCE = _history_source()
    OBS_CONFIG = _obs_config()
    TIME_MAP = _time_map()
    # ************* Model config  *************

    # ************* Ensemble config  *************
    _gen_param_key = ResPrototype("char* config_keys_get_gen_param_key()", bind=False)
    GEN_PARAM = _gen_param_key()
    _forward_init_key = ResPrototype(
        "char* config_keys_get_forward_init_key()", bind=False
    )
    FORWARD_INIT = _forward_init_key()
    _min_std_key = ResPrototype("char* config_keys_get_min_std_key()", bind=False)
    MIN_STD = _min_std_key()
    _key_key = ResPrototype("char* config_keys_get_key_key()", bind=False)
    KEY_KEY = _key_key()
    _kw_tag_format_key = ResPrototype(
        "char* config_keys_get_kw_tag_format_key()", bind=False
    )
    GEN_KW_TAG_FORMAT = _kw_tag_format_key()
    _surface_key = ResPrototype("char* config_keys_get_surface_key()", bind=False)
    SURFACE_KEY = _surface_key()
    _base_surface_key = ResPrototype(
        "char* config_keys_get_base_surface_key()", bind=False
    )
    BASE_SURFACE_KEY = _base_surface_key()
    _field_key = ResPrototype("char* config_keys_get_field_key()", bind=False)
    FIELD_KEY = _field_key()
    VAR_TYPE = "VAR_TYPE"
    _init_transform = ResPrototype(
        "char* config_keys_get_init_transform_key()", bind=False
    )
    INIT_TRANSFORM = _init_transform()
    _input_transform = ResPrototype(
        "char* config_keys_get_input_transform_key()", bind=False
    )
    INPUT_TRANSFORM = _input_transform()
    _output_transform = ResPrototype(
        "char* config_keys_get_output_transform_key()", bind=False
    )
    OUTPUT_TRANSFORM = _output_transform()
    _min_key = ResPrototype("char* config_keys_get_min_key()", bind=False)
    MIN_KEY = _min_key()
    _max_key = ResPrototype("char* config_keys_get_max_key()", bind=False)
    MAX_KEY = _max_key()
    ENKF_INFILE = "ENKF_INFILE"
    _parameter_key = ResPrototype("char* config_keys_get_parameter_key()", bind=False)
    PARAMETER_KEY = _parameter_key()
    _general_key = ResPrototype("char* config_keys_get_general_key()", bind=False)
    GENERAL_KEY = _general_key()
    _pred_key = ResPrototype("char* config_keys_get_pred_key()", bind=False)
    PRED_KEY = _pred_key()
    _container_key = ResPrototype("char* config_keys_get_container_key()", bind=False)
    CONTAINER_KEY = _container_key()
    # ************* Ensemble config  *************

    SLURM_SBATCH_OPTION = _sbatch_option()
    SLURM_SCANCEL_OPTION = _scancel_option()
    SLURM_SCONTROL_OPTION = _scontrol_option()
    SLURM_SQUEUE_OPTION = _squeue_option()
    SLURM_PARTITION_OPTION = _partition_option()
    SLURM_SQUEUE_TIMEOUT_OPTION = _squeue_timeout_option()
    SLURM_MAX_RUNTIME_OPTION = _max_runtime_option()
    SLURM_MEMORY_OPTION = _memory_option()
    SLURM_MEMORY_PER_CPU_OPTION = _memory_per_cpu_option()
    SLURM_EXCLUDE_HOST_OPTION = _exclude_host_option()
    SLURM_INCLUDE_HOST_OPTION = _include_host_option()
