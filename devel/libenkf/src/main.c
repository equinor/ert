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
#include <ext_joblist.h>
#include <enkf_sched.h>
#include <stringlist.h>


void enkf_sched_test(const enkf_main_type * enkf_main , const ext_joblist_type * joblist) {
  stringlist_type * forward_model = stringlist_alloc_argv_ref((const char *[3]) {"REPLERM" , "ECLIPSE100" , "SEISMIC_TEST"} , 3);
  enkf_sched_type * enkf_sched = enkf_sched_fscanf_alloc( "sched_config" , enkf_main_get_sched_file( enkf_main ) , joblist , forward_model);
  enkf_sched_fprintf(enkf_sched , stdout);
  enkf_sched_free( enkf_sched );
  stringlist_free( forward_model );
}

void install_SIGNALS(void) {
  signal(SIGSEGV , util_abort_signal);
}

void enkf_welcome() {
  printf("\n");
  printf("svn version......: %s \n",SVN_VERSION);
  printf("Compile time.....: %s \n",COMPILE_TIME_STAMP);
  printf("\n");
}


void enkf_usage() {
  printf("\n");
  printf(" *********************************************************************\n");
  printf(" **                       E n K F                                   **\n");
  printf(" **                                                                 **\n");
  printf(" **-----------------------------------------------------------------**\n");
  printf(" ** You have sucessfully started the EnKF program developed at      **\n");
  printf(" ** StatoilHydro. Before you can actually start using the program,  **\n");
  printf(" ** you must create a configuration file. When the configuration    **\n");
  printf(" ** file has been created, you can start the enkf application with: **\n");
  printf(" **                                                                 **\n");
  printf(" **   bash> enkf config_file                                        **\n");
  printf(" **                                                                 **\n");
  printf(" ** Instructions on how to create the configuration file can be     **\n");
  printf(" ** found at: http://sdp.statoil.no/wiki/index.php/Res:Setup-EnKF   **\n");
  printf(" *********************************************************************\n");
}


int main (int argc , char ** argv) {
  enkf_welcome();
  install_SIGNALS();
  if (argc != 2) {
    enkf_usage();
    exit(1);
  } else {
    const char * site_config_file = SITE_CONFIG_FILE;  /* The variable SITE_CONFIG_FILE should be defined on compilation ... */
    const char * config_file      = argv[1];
    ext_joblist_type * joblist;
    job_queue_type   * job_queue;
    enkf_main_type   * enkf_main;
    enkf_site_config_type * site_config = enkf_site_config_bootstrap(site_config_file);
    joblist   = ext_joblist_alloc();

    enkf_config_type  * enkf_config              = enkf_config_fscanf_alloc(config_file , site_config , joblist , 1 , false , false , true);
    plain_driver_type * dynamic_analyzed 	 = plain_driver_alloc(enkf_config_get_ens_path(enkf_config) 	      , "%04d/mem%03d/Analyzed");
    plain_driver_type * dynamic_forecast 	 = plain_driver_alloc(enkf_config_get_ens_path(enkf_config) 	      , "%04d/mem%03d/Forecast");
    plain_driver_parameter_type * parameter      = plain_driver_parameter_alloc(enkf_config_get_ens_path(enkf_config) , "%04d/mem%03d/Parameter");
    plain_driver_type * eclipse_static           = plain_driver_alloc(enkf_config_get_ens_path(enkf_config)           , "%04d/mem%03d/Static");
    fs_index_type     * fs_index                 = fs_index_alloc(enkf_config_get_ens_path(enkf_config)               , "%04d/mem%03d/INDEX");
    enkf_fs_type      * fs = enkf_fs_alloc(fs_index , dynamic_analyzed, dynamic_forecast , eclipse_static , parameter);


    plain_driver_README( enkf_config_get_ens_path(enkf_config) );
    job_queue = enkf_config_alloc_job_queue(enkf_config , site_config);
    enkf_main = enkf_main_alloc(enkf_config , fs , job_queue , joblist);
    const enkf_sched_type * enkf_sched = enkf_sched_fscanf_alloc( enkf_config_get_enkf_sched_file(enkf_config) , enkf_main_get_sched_file(enkf_main) , joblist , enkf_config_get_forward_model(enkf_config));
    enkf_main_initialize_ensemble(enkf_main); 
    
    
    {
      
      bool unlink_run_path = true;
      const int num_nodes            = enkf_sched_get_num_nodes(enkf_sched);
      const int schedule_num_reports = enkf_sched_get_schedule_num_reports(enkf_sched);
      int inode;
      enkf_sched_fprintf(enkf_sched, stdout);
      for (inode = 0; inode < num_nodes; inode++) {
	const enkf_sched_node_type * node = enkf_sched_iget_node(enkf_sched , inode);
	int report_step1;
	int report_step2;
	int report_stride;
	int report_step;
	int next_report_step;
	bool enkf_on;
	stringlist_type * forward_model;

	enkf_sched_node_get_data(node , &report_step1 , &report_step2 , &report_stride , &enkf_on , &forward_model);
	report_step = report_step1;
	do {
	  next_report_step = util_int_min(schedule_num_reports , util_int_min(report_step + report_stride , report_step2));
	  printf("Running: %d -> %d (%d) \n",report_step , next_report_step, report_stride);

	  enkf_main_run(enkf_main , report_step , report_step , next_report_step , enkf_on , unlink_run_path , forward_model);
	  report_step = next_report_step;
	} while (next_report_step < report_step2);
      }
    }
    
    job_queue_free(job_queue);
    enkf_main_free(enkf_main);
    enkf_site_config_free(site_config); /* Should probably be owned by enkf_main ?? */

    ext_joblist_free(joblist);
    enkf_fs_free(fs);  /* Takes the drivers as well */
  }
}
