#include <enkf_fs.h>
#include <enkf_main.h>
#include <util.h>
#include <config.h>
#include <hash.h>
#include <enkf_types.h>
#include <string.h>
#include <local_driver.h>
#include <lsf_driver.h>
#include <signal.h>
#include <ext_joblist.h>
#include <enkf_sched.h>
#include <stringlist.h>
#include <enkf_tui_main.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <local_config.h>


void install_SIGNALS(void) {
  signal(SIGSEGV , util_abort_signal);    /* Segmentation violation, i.e. overwriting memory ... */
  signal(SIGINT  , util_abort_signal);    /* Control C */
  signal(SIGTERM , util_abort_signal);    /* If killing the enkf program with SIGTERM (the default kill signal) you will get a backtrace. Killing with SIGKILL (-9) will not give a backtrace.*/
}


void text_splash() {
  const int usleep_time = 1250;
  int i;
  {
    //#include "statoilhydro.h"
#include "ERT.h"
    printf("\n\n");
    for (i = 0; i < SPLASH_LENGTH; i++) {
      printf("%s\n" , splash_text[i]);
      usleep(usleep_time);
    }
    printf("\n\n");

    sleep(1);
#undef SPLASH_LENGTH
  }
}


/*
  SVN_VERSION and COMPILE_TIME_STAMP are env variables set by the
  makefile. Will exit if the config file does not exist.
*/
void enkf_welcome(const char * config_file) {
  if (util_file_exists( config_file )) {
    char * svn_version  	 = util_alloc_sprintf("svn version..........: %s \n",SVN_VERSION);
    char * compile_time 	 = util_alloc_sprintf("Compile time.........: %s \n",COMPILE_TIME_STAMP);
    char * abs_path     	 = util_alloc_realpath( config_file );
    char * config_file_msg       = util_alloc_sprintf("Configuration file...: %s \n",abs_path);
    
    /* This will be printed if/when util_abort() is called on a later stage. */
    util_abort_append_version_info(svn_version);
    util_abort_append_version_info(compile_time);
    util_abort_append_version_info(config_file_msg);
    
    free(config_file_msg);
    free(abs_path);
    free(svn_version);
    free(compile_time);
  } else util_exit(" ** Sorry: can not locate configuration file: %s \n\n" , config_file);
}


void enkf_usage() {
  printf("\n");
  printf(" *********************************************************************\n");
  printf(" **                                                                 **\n");
  printf(" **                            E R T                                **\n");
  printf(" **                                                                 **\n");
  printf(" **-----------------------------------------------------------------**\n");
  printf(" ** You have sucessfully started the ert program developed at       **\n");
  printf(" ** StatoilHydro. Before you can actually start using the program,  **\n");
  printf(" ** you must create a configuration file. When the configuration    **\n");
  printf(" ** file has been created, you can start the ert application with:  **\n");
  printf(" **                                                                 **\n");
  printf(" **   bash> ert config_file                                         **\n");
  printf(" **                                                                 **\n");
  printf(" ** Instructions on how to create the configuration file can be     **\n");
  printf(" ** found at: http://sdp.statoil.no/wiki/index.php/res:enkf         **\n");
  printf(" *********************************************************************\n");
}






int main (int argc , char ** argv) {
  text_splash();
  printf("\n");
  printf("svn version : %s \n",SVN_VERSION);
  printf("compile time: %s \n",COMPILE_TIME_STAMP);
  install_SIGNALS();
  if (argc != 2) {
    enkf_usage();
    exit(1);
  } else {
    const char * site_config_file  = SITE_CONFIG_FILE;  /* The variable SITE_CONFIG_FILE should be defined on compilation ... */
    const char * model_config_file = argv[1]; 
    
    enkf_welcome( model_config_file );
    {
      enkf_main_type * enkf_main = enkf_main_bootstrap(site_config_file , model_config_file);
      printf("Bootstrap complete \n");
      /*****************************************************************/
      /* Test code of various kinds can be added here */

      {
        site_config_type * site_config = enkf_main_get_site_config( enkf_main );
        
        printf("max_running_LOCAL: %d \n", site_config_get_max_running_local( site_config ));
        printf("max_running_RSH:   %d \n", site_config_get_max_running_rsh( site_config ));

        site_config_set_max_running_rsh( site_config , 10 );
        site_config_set_max_running_local( site_config , 77 );
        
        printf("max_running_LOCAL: %d \n", site_config_get_max_running_local( site_config ));
        printf("max_running_RSH:   %d \n", site_config_get_max_running_rsh( site_config ));
      }
      

      /*****************************************************************/
      enkf_main_free(enkf_main);
    }
    util_abort_free_version_info(); /* No fucking leaks ... */
  }
}
