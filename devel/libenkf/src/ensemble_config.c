#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <stdbool.h>
#include <unistd.h>
#include <util.h>
#include <hash.h>
#include <set.h>
#include <multz_config.h>
#include <enkf_config_node.h>
#include <path_fmt.h>
#include <enkf_types.h>
#include <well_config.h>
#include <field_config.h>
#include <equil_config.h>
#include <gen_param_config.h>
#include <multflt_config.h>
#include <well_obs.h>
#include <thread_pool.h>
#include <obs_node.h>
#include <obs_data.h>
#include <meas_matrix.h>
#include <enkf_types.h>
#include <analysis.h>
#include <enkf_obs.h>
#include <sched_file.h>
#include <enkf_fs.h>
#include <void_arg.h>
#include <gen_kw_config.h>
#include <enkf_config.h>
#include <ecl_grid.h>
#include <time.h>
#include <enkf_site_config.h>
#include <job_queue.h>
#include <lsf_driver.h>
#include <local_driver.h>
#include <rsh_driver.h>
#include <relperm_config.h>
#include <summary.h>
#include <summary_config.h>
#include <havana_fault_config.h>
#include <ext_joblist.h>
#include <gen_data.h>
#include <gen_data_config.h>
#include <stringlist.h>
#include <ensemble_config.h>


struct ensemble_config_struct {
  int  		   ens_size;
  hash_type       *config_nodes;
};



ensemble_config_type * ensemble_config_alloc(int ens_size) {
  if (ens_size <= 0)
    util_exit("%s: ensemble size must be > 0 \n",__func__);
  {
    ensemble_config_type * ensemble_config = util_malloc(sizeof * ensemble_config , __func__);
    ensemble_config->ens_size     = ens_size;
    ensemble_config->config_nodes = hash_alloc();
    return ensemble_config;
  }
}



void ensemble_config_free(ensemble_config_type * ensemble_config) {
  hash_free(ensemble_config->config_nodes);
  free(ensemble_config);
}
