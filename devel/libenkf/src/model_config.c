#include <util.h>
#include <stdlib.h>
#include <enkf_fs.h>
#include <time.h>
#include <path_fmt.h>
#include <enkf_sched.h>
#include <model_config.h>
#include <history.h>
#include <config.h>
#include <sched_file.h>
#include <ecl_sum.h>
#include <ecl_util.h>


/**
   This struct contains configuration which is specific to this
   particular model/run. Much of the information is actually accessed
   directly through the enkf_state object; but this struct is the
   owner of the information, and responsible for allocating/freeing
   it.

   Observe that the distinction of what goes in model_config, and what
   goes in ecl_config is not entirely clear; ECLIPSE is unfortunately
   not (yet ??) exactly 'any' reservoir simulator in this context.
*/


struct model_config_struct {
  enkf_fs_type      * ensemble_dbase;     /* Where the ensemble files are stored */
  
  history_type      * history;            /* The history object. */
  time_t              start_date;         /* When the history starts. */ 
  
  stringlist_type   * forward_model;      /* A list of external jobs - which acts as keys into a ext_joblist_type instance. */
  path_fmt_type     * result_path;        /* path_fmt instance for results - should contain one %d which will be replaced report_step */
  path_fmt_type     * runpath;            /* path_fmt instance for runpath - runtime the call gets arguments: (iens, report_step1 , report_step2) - i.e. at least one %d must be present.*/  
  enkf_sched_type   * enkf_sched;         /* The enkf_sched object controlling when the enkf is ON|OFF, strides in report steps and special forward model. */
};






model_config_type * model_config_alloc(const config_type * config , time_t start_date) {
  model_config_type * model_config = util_malloc(sizeof * model_config , __func__);
  model_config->start_date    = start_date;
  model_config->result_path   = path_fmt_alloc_directory_fmt( config_get(config , "RESULT_PATH") );
  model_config->runpath       = path_fmt_alloc_directory_fmt( config_get(config , "RUNPATH") );
  model_config->forward_model = config_alloc_stringlist( config , "FORWARD_MODEL" );
  
  /*enkf_sched_fscanf_alloc( enkf_config_get_enkf_sched_file(enkf_config) , enkf_main_get_sched_file(enkf_main) , joblist , enkf_config_get_forward_model(enkf_config));*/
  model_config->enkf_sched  = NULL;
  return model_config;
}



void model_config_free(model_config_type * model_config) {
  path_fmt_free(  model_config->result_path );
  path_fmt_free(  model_config->runpath );
  /*enkf_sched_free( model_config->enkf_sched );*/
  free(model_config);
}
