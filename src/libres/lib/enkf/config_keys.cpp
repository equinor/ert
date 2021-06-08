/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'config_keys.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#include <ert/enkf/config_keys.hpp>

const char * config_keys_get_config_directory_key() {
  return CONFIG_DIRECTORY_KEY;
}

const char * config_keys_get_config_file_key() {
  return RES_CONFIG_FILE_KEY;
}

const char * config_keys_get_queue_system_key() {
  return QUEUE_SYSTEM_KEY;
}

const char * config_keys_get_run_template_key() {
  return RUN_TEMPLATE_KEY;
}

const char * config_keys_get_gen_kw_key() {
  return GEN_KW_KEY;
}

const char * config_keys_get_queue_option_key() {
  return QUEUE_OPTION_KEY;
}

const char * config_keys_get_install_job_key() {
  return INSTALL_JOB_KEY;
}

const char * config_keys_get_log_file_key() {
  return LOG_FILE_KEY;
}

const char * config_keys_get_log_level_key() {
  return LOG_LEVEL_KEY;
}

const char * config_keys_get_update_log_path_key() {
  return UPDATE_LOG_PATH_KEY;
}

const char * config_keys_get_summary_key() {
  return SUMMARY_KEY;
}

const char * config_keys_get_max_runtime_key() {
  return MAX_RUNTIME_KEY;
}

const char * config_keys_get_min_realizations_key() {
  return MIN_REALIZATIONS_KEY;
}

const char * config_keys_get_max_submit_key() {
  return MAX_SUBMIT_KEY;
}

const char * config_keys_get_umask_key() {
  return UMASK_KEY;
}

const char * config_keys_get_data_kw_key() {
    return DATA_KW_KEY;
}

const char * config_keys_get_runpath_file_key() {
  return RUNPATH_FILE_KEY;
}

const char * config_keys_get_gen_data_key() {
  return GEN_DATA_KEY;
}

/* ************* ECL config  ************* */
const char * config_keys_get_eclbase_key() {
  return ECLBASE_KEY;
}

const char * config_keys_get_data_file_key() {
  return DATA_FILE_KEY;
}

const char * config_keys_get_grid_key() {
  return GRID_KEY;
}

const char * config_keys_get_refcase_key() {
  return REFCASE_KEY;
}

const char * config_keys_get_refcase_list_key() {
  return REFCASE_LIST_KEY;
}

const char * config_keys_get_end_date_key() {
  return END_DATE_KEY;
}

const char * config_keys_get_schedule_prediction_file_key() {
  return SCHEDULE_PREDICTION_FILE_KEY;
}
/* ************* ECL config  ************* */

const char * config_keys_get_result_file() {
  return RESULT_FILE_KEY;
}

const char * config_keys_get_report_steps() {
  return REPORT_STEPS_KEY;
}

const char * config_keys_get_input_format() {
  return INPUT_FORMAT_KEY;
}

const char * config_keys_get_ecl_file() {
  return ECL_FILE_KEY;
}

const char * config_keys_get_output_format() {
  return OUTPUT_FORMAT_KEY;
}

const char * config_keys_get_init_files() {
  return INIT_FILES_KEY;
}

const char * config_keys_get_random_seed() {
  return RANDOM_SEED_KEY;
}
/////***** Analysis config keys*******////////////
const char * config_keys_get_alpha() {
    return ENKF_ALPHA_KEY;
}

const char * config_keys_get_std_cutoff() {
    return STD_CUTOFF_KEY;
}
const char * config_keys_get_stop_long_running() {
    return STOP_LONG_RUNNING_KEY;
}

const char * config_keys_get_single_node_update() {
    return SINGLE_NODE_UPDATE_KEY;
}

const char * config_keys_get_std_scale_correlated_obs() {
    return STD_SCALE_CORRELATED_OBS_KEY;
}

const char * config_keys_get_rerun() {
    return ENKF_RERUN_KEY;
}

const char * config_keys_get_rerun_start() {
    return RERUN_START_KEY;
}

const char * config_keys_get_merge_observations() {
    return ENKF_MERGE_OBSERVATIONS_KEY;
}

const char * config_keys_get_analysis_load() {
    return ANALYSIS_LOAD_KEY;
}

const char * config_keys_get_analysis_copy() {
    return ANALYSIS_COPY_KEY;
}

const char * config_keys_get_analysis_select() {
    return ANALYSIS_SELECT_KEY;
}

const char * config_keys_get_analysis_set_var() {
    return ANALYSIS_SET_VAR_KEY;
}
/////***** Analysis config keys*******////////////
const char * config_keys_get_install_job_directory_key() {
  return INSTALL_JOB_DIRECTORY_KEY;
}

const char * config_keys_get_license_path_key() {
  return LICENSE_PATH_KEY;
}

const char * config_keys_get_setenv_key() {
  return SETENV_KEY;
}

const char * config_keys_get_job_script_key() {
  return JOB_SCRIPT_KEY;
}

const char * config_keys_get_num_cpu_key() {
  return NUM_CPU_KEY;
}

const char * config_keys_get_define_key() {
    return DEFINE_KEY;
}

//*********analysis_iter_config keys************//
const char * config_keys_get_iter_case_key() {
    return ITER_CASE_KEY;
}

const char * config_keys_get_iter_count_key() {
    return ITER_COUNT_KEY;
}

const char * config_keys_get_iter_retry_count_key() {
    return ITER_RETRY_COUNT_KEY;
}
//*********analysis_iter_config keys************//

//*********ert workflow list_config keys************//
const char * config_keys_get_load_workflow_key() {
  return LOAD_WORKFLOW_KEY;
}

const char * config_keys_get_load_workflow_job_key() {
  return LOAD_WORKFLOW_JOB_KEY;
}

const char * config_keys_get_workflow_job_directory_key() {
  return WORKFLOW_JOB_DIRECTORY_KEY;
}
//*********ert workflow list_config keys************//

const char * config_keys_get_hook_workflow_key() {
  return HOOK_WORKFLOW_KEY;
}
// hook_manager config keys

/* ************* Model config  ************* */

const char * config_keys_get_max_resample_key() {
  return MAX_RESAMPLE_KEY;
}

const char * config_keys_get_num_realizations_key() {
  return NUM_REALIZATIONS_KEY;
}

const char * config_keys_get_runpath_key() {
  return RUNPATH_KEY;
}

const char * config_keys_get_data_root_key() {
  return DATA_ROOT_KEY;
}

const char * config_keys_get_enspath_key() {
  return ENSPATH_KEY;
}

const char * config_keys_get_jobname_key() {
  return JOBNAME_KEY;
}

const char * config_keys_get_forward_model_key() {
  return FORWARD_MODEL_KEY;
}

const char * config_keys_get_simulation_job_key() {
  return SIMULATION_JOB_KEY;
}

const char * config_keys_get_rftpath_key() {
  return RFTPATH_KEY;
}

const char * config_keys_get_gen_kw_export_name_key() {
  return GEN_KW_EXPORT_NAME_KEY;
}

const char * config_keys_get_history_source_key() {
  return HISTORY_SOURCE_KEY;
}

const char * config_keys_get_obs_config_key() {
  return OBS_CONFIG_KEY;
}

const char * config_keys_get_time_map_key() {
  return TIME_MAP_KEY;
}

/* ************* Model config  ************* */

/* ************* Ensemble config  ************* */
  const char * config_keys_get_gen_param_key() {
    return GEN_PARAM_KEY;
  }

  const char * config_keys_get_forward_init_key() {
    return FORWARD_INIT_KEY;
  }

  const char * config_keys_get_min_std_key() {
    return MIN_STD_KEY;
  }

  const char * config_keys_get_template_key() {
    return TEMPLATE_KEY;
  }

  const char * config_keys_get_key_key() {
    return KEY_KEY;
  }

  const char * config_keys_get_kw_tag_format_key() {
    return GEN_KW_TAG_FORMAT_KEY;
  }

  const char * config_keys_get_surface_key() {
    return SURFACE_KEY;
  }

  const char * config_keys_get_base_surface_key() {
    return BASE_SURFACE_KEY;
  }

  const char * config_keys_get_field_key() {
    return FIELD_KEY;
  }

  const char * config_keys_get_init_transform_key() {
    return INIT_TRANSFORM_KEY;
  }

  const char * config_keys_get_input_transform_key() {
    return INPUT_TRANSFORM_KEY;
  }

  const char * config_keys_get_output_transform_key() {
    return OUTPUT_TRANSFORM_KEY;
  }

  const char * config_keys_get_min_key() {
    return MIN_KEY;
  }

  const char * config_keys_get_max_key() {
    return MAX_KEY;
  }

  const char * config_keys_get_parameter_key() {
    return PARAMETER_KEY;
  }

  const char * config_keys_get_general_key() {
    return GENERAL_KEY;
  }

  const char * config_keys_get_pred_key() {
    return PRED_KEY;
  }

  const char * config_keys_get_container_key() {
    return CONTAINER_KEY;
  }

  const char * config_keys_get_slurm_sbatch_option() {
    return SLURM_SBATCH_OPTION;
  }

  const char * config_keys_get_slurm_scancel_option() {
    return SLURM_SCANCEL_OPTION;
  }

  const char * config_keys_get_slurm_scontrol_option() {
    return SLURM_SCONTROL_OPTION;
  }

  const char * config_keys_get_slurm_squeue_option() {
    return SLURM_SQUEUE_OPTION;
  }

  const char * config_keys_get_slurm_partition_option() {
    return SLURM_PARTITION_OPTION;
  }

  const char * config_keys_get_slurm_squeue_timeout_option() {
    return SLURM_SQUEUE_TIMEOUT_OPTION;
  }

  const char * config_keys_get_slurm_max_runtime_option() {
    return SLURM_MAX_RUNTIME_OPTION;
  }

  const char * config_keys_get_slurm_memory_option() {
    return SLURM_MEMORY_OPTION;
  }

  const char * config_keys_get_slurm_memory_per_cpu_option() {
    return SLURM_MEMORY_PER_CPU_OPTION;
  }

  const char * config_keys_get_slurm_exclude_host_option() {
    return SLURM_EXCLUDE_HOST_OPTION;
  }

  const char * config_keys_get_slurm_include_host_option() {
    return SLURM_INCLUDE_HOST_OPTION;
  }



/* ************* Ensemble config  ************* */
