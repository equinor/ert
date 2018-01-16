#  Copyright (C) 2017  Statoil ASA, Norway.
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

from res.enkf import EnkfPrototype

class ConfigKeys:

    _config_directory_key = EnkfPrototype("char* config_keys_get_config_directory_key()", bind=False)
    _queue_system_key     = EnkfPrototype("char* config_keys_get_queue_system_key()", bind=False)
    _run_template_key     = EnkfPrototype("char* config_keys_get_run_template_key()", bind=False)
    _gen_kw_key           = EnkfPrototype("char* config_keys_get_gen_kw_key()", bind=False)
    _history_source_key   = EnkfPrototype("char* config_keys_get_history_source_key()", bind=False)
    _queue_option_key     = EnkfPrototype("char* config_keys_get_queue_option_key()", bind=False)
    _install_job_key      = EnkfPrototype("char* config_keys_get_install_job_key()", bind=False)
    _plot_settings_key    = EnkfPrototype("char* config_keys_get_plot_setting_key()", bind=False)
    _forward_model_key    = EnkfPrototype("char* config_keys_get_forward_model_key()", bind=False)
    _simulation_job_key   = EnkfPrototype("char* config_keys_get_simulation_job_key()", bind=False)
    _log_file_key         = EnkfPrototype("char* config_keys_get_log_file_key()", bind=False)
    _log_level_key        = EnkfPrototype("char* config_keys_get_log_level_key()", bind=False)
    _update_log_path      = EnkfPrototype("char* config_keys_get_update_log_path_key()", bind=False)
    _store_seed           = EnkfPrototype("char* config_keys_get_store_seed_key()", bind=False)
    _load_seed            = EnkfPrototype("char* config_keys_get_load_seed_key()", bind=False)
    _summary              = EnkfPrototype("char* config_keys_get_summary_key()", bind=False)
    _jobname              = EnkfPrototype("char* config_keys_get_jobname_key()", bind=False)
    _max_runtime          = EnkfPrototype("char* config_keys_get_max_runtime_key()", bind=False)
    _min_realizations     = EnkfPrototype("char* config_keys_get_min_realizations_key()", bind=False)
    _max_submit           = EnkfPrototype("char* config_keys_get_max_submit_key()", bind=False)
    _umask                = EnkfPrototype("char* config_keys_get_umask_key()", bind=False)
    _data_file            = EnkfPrototype("char* config_keys_get_data_file_key()", bind=False)
    _runpath              = EnkfPrototype("char* config_keys_get_runpath_key()", bind=False)
    _runpath_file         = EnkfPrototype("char* config_keys_get_runpath_file_key()", bind=False)
    _eclbase              = EnkfPrototype("char* config_keys_get_eclbase_key()", bind=False)
    _num_realizations     = EnkfPrototype("char* config_keys_get_num_realizations_key()", bind=False)
    _enspath              = EnkfPrototype("char* config_keys_get_enspath_key()", bind=False)
    _grid                 = EnkfPrototype("char* config_keys_get_grid_key()", bind=False)
    _refcase              = EnkfPrototype("char* config_keys_get_refcase_key()", bind=False)
    _history_source       = EnkfPrototype("char* config_keys_get_history_source_key()", bind=False)
    _obs_config           = EnkfPrototype("char* config_keys_get_obs_config_key()", bind=False)
    _time_map             = EnkfPrototype("char* config_keys_get_time_map_key()", bind=False)
    _gen_data             = EnkfPrototype("char* config_keys_get_gen_data_key()", bind=False)
    _result_file          = EnkfPrototype("char* config_keys_get_result_file()", bind=False)
    _report_steps         = EnkfPrototype("char* config_keys_get_report_steps()", bind=False)
    _input_format         = EnkfPrototype("char* config_keys_get_input_format()", bind=False)
    _ecl_file             = EnkfPrototype("char* config_keys_get_ecl_file()", bind=False)
    _output_format        = EnkfPrototype("char* config_keys_get_output_format()", bind=False)
    _init_files           = EnkfPrototype("char* config_keys_get_init_files()", bind=False)
    _random_seed          = EnkfPrototype("char* config_keys_get_random_seed()", bind=False)

    ARGLIST          = "ARGLIST"
    CONFIG_DIRECTORY = _config_directory_key()
    DEFINES          = "DEFINES"
    INTERNALS        = "INTERNALS"
    SIMULATION       = "SIMULATION"
    LOGGING          = "LOGGING"
    SEED             = "SEED"
    QUEUE_SYSTEM     = _queue_system_key()
    RUN_TEMPLATE     = _run_template_key()
    TEMPLATE         = "TEMPLATE"
    EXPORT           = "EXPORT"
    GEN_KW           = _gen_kw_key()
    NAME             = "NAME"
    OUT_FILE         = "OUT_FILE"
    PARAMETER_FILE   = "PARAMETER_FILE"
    PATH             = "PATH"
    QUEUE_OPTION     = _queue_option_key()
    DRIVER_NAME      = "DRIVER_NAME"
    OPTION           = "OPTION"
    VALUE            = "VALUE"
    INSTALL_JOB      = _install_job_key()
    PLOT_SETTINGS    = _plot_settings_key()
    FORWARD_MODEL    = _forward_model_key()
    SIMULATION_JOB   = _simulation_job_key()
    LOG_FILE         = _log_file_key()
    LOG_LEVEL        = _log_level_key()
    UPDATE_LOG_PATH  = _update_log_path()
    STORE_SEED       = _store_seed()
    LOAD_SEED        = _load_seed()
    RANDOM_SEED      = _random_seed()
    SUMMARY          = _summary()
    JOBNAME          = _jobname()
    MAX_RUNTIME      = _max_runtime()
    MIN_REALIZATIONS = _min_realizations()
    MAX_SUBMIT       = _max_submit()
    UMASK            = _umask()
    MAX_RUNNING      = "MAX_RUNNING"
    DATA_FILE        = _data_file()
    RUNPATH          = _runpath()
    RUNPATH_FILE     = _runpath_file()
    ECLBASE          = _eclbase()
    NUM_REALIZATIONS = _num_realizations()
    ENSPATH          = _enspath()
    GRID             = _grid()
    REFCASE          = _refcase()
    HISTORY_SOURCE   = _history_source()
    OBS_CONFIG       = _obs_config()
    TIME_MAP         = _time_map()
    GEN_DATA         = _gen_data()
    RESULT_FILE      = _result_file()
    REPORT_STEPS     = _report_steps()
    INPUT_FORMAT     = _input_format()
    ECL_FILE         = _ecl_file()
    OUTPUT_FORMAT    = _output_format()
    INIT_FILES       = _init_files()
