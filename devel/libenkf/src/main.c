#include <enkf_fs.h>
#include <enkf_main.h>
#include <enkf_config.h>
#include <enkf_site_config.h>
#include <util.h>
#include <basic_queue_driver.h>
#include <plain_driver.h>
#include <plain_driver_parameter.h>
#include <config.h>
#include <hash.h>
#include <fs_index.h>
#include <enkf_types.h>
#include <string.h>
#include <local_driver.h>
#include <lsf_driver.h>
#include <signal.h>


void install_SIGNALS(void) {
  signal(SIGSEGV , util_abort_signal);
}




int main (int argc , char ** argv) {

  install_SIGNALS();
  if (argc != 2) 
    util_exit("%s: usage %s config_file \n",__func__ , argv[0]);
  else {
    const char * site_config_file = SITE_CONFIG_FILE;  /* The variable SITE_CONFIG_FILE should be defined on compilation ... */
    const char * config_file      = argv[1];
    ecl_queue_type   * ecl_queue;
    enkf_main_type   * enkf_main;
    enkf_site_config_type * site_config = enkf_site_config_bootstrap(site_config_file);
    enkf_config_type * enkf_config      = enkf_config_fscanf_alloc(config_file , site_config , 1 , false , false , true);
    
    plain_driver_type * dynamic_analyzed 	 = plain_driver_alloc(enkf_config_get_ens_path(enkf_config) 	   , "%04d/mem%03d/Analyzed");
    plain_driver_type * dynamic_forecast 	 = plain_driver_alloc(enkf_config_get_ens_path(enkf_config) 	   , "%04d/mem%03d/Forecast");
    plain_driver_parameter_type * parameter      = plain_driver_parameter_alloc(enkf_config_get_ens_path(enkf_config) , "%04d/mem%03d/Parameter");
    plain_driver_type * eclipse_static           = plain_driver_alloc(enkf_config_get_ens_path(enkf_config)    , "%04d/mem%03d/Static");
    fs_index_type     * fs_index                 = fs_index_alloc(enkf_config_get_ens_path(enkf_config) , "INDEX/mem%03d/INDEX");
    enkf_fs_type      * fs = enkf_fs_alloc(fs_index , dynamic_analyzed, dynamic_forecast , eclipse_static , parameter);
    
    ecl_queue = enkf_config_alloc_ecl_queue(enkf_config , site_config);
    enkf_main = enkf_main_alloc(enkf_config , fs , ecl_queue);

    enkf_main_initialize_ensemble(enkf_main); 
    
    {
      int report_step;
      
      for (report_step = 0; report_step < 61; report_step++)
	enkf_main_run(enkf_main , report_step , report_step + 1 , true);
    }
    
    ecl_queue_free(ecl_queue);
    enkf_main_free(enkf_main);
    enkf_site_config_free(site_config); /* Should probably be owned by enkf_main ?? */

    enkf_fs_free(fs);  /* Takes the drivers as well */
  }
}
