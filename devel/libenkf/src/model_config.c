#include <util.h>
#include <stdlib.h>
#include <enkf_fs.h>
#include <time.h>
#include <path_fmt.h>
#include <enkf_sched.h>
#include <model_config.h>
#include <history.h>



/**
   This struct contains configuration which is specific to this
   particular model/run. Much of the information is actually accessed
   directly through the enkf_state object; but this struct is the
   owner of the information, and responsible for allocating/freeing
   it.

   Observe that the distinction of what goes in model_config, and what 
   goes in ecl_config is not entirely clear.
*/

struct model_config_struct {
  enkf_fs_type      * ensemble_dbase;
  
  history_type      * history;
  time_t              start_date;
  
  
  path_fmt_type     * result_path;
  path_fmt_type     * runpath;
  enkf_sched_type   * enkf_sched;
};


