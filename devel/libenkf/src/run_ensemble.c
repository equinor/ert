#include <enkf_fs.h>
#include <enkf_main.h>
#include <util.h>
#include <config.h>
#include <hash.h>
//#include <fs_index.h>
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
#include <ensemble_config.h>


void install_SIGNALS(void) {
  signal(SIGSEGV , util_abort_signal);
  signal(SIGINT  , util_abort_signal);
  signal(SIGKILL , util_abort_signal);
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
  printf(" **                                                                 **\n");
  printf(" **                           E n O P T                             **\n");
  printf(" **                                                                 **\n");
  printf(" *********************************************************************\n");
}



int main (int argc , char ** argv) {
  enkf_welcome();
  install_SIGNALS();

  if (argc != 3) {
    enkf_usage();
    exit(1);
  } else {
    const char * site_config_file  = SITE_CONFIG_FILE;  /* The variable SITE_CONFIG_FILE should be defined on compilation ... */
    const char * model_config_file = argv[1]; 
    const int enopt_start = atoi(argv[2]);
    const bool load_results = false;    
    
    enkf_main_type * enkf_main = enkf_main_bootstrap(site_config_file , model_config_file);
    
    
    {
      const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
      const int ens_size                           = ensemble_config_get_size(ensemble_config);
      const int enopt_stop = enopt_start + 1 ;

      printf("ENS_SIZE: %d\n", ens_size);
      //const int enopt_start = util_scanf_int("Restart Step",argv[2]);
      //const int ens_size    = ensemble_config_get_size(ensemble_config);
      bool * iactive        = util_malloc(ens_size * sizeof * iactive , __func__);
      {
        int iens;
        for (iens= 0; iens < ens_size; iens++)
          iactive[iens] = true;
      }
      
      // Joakim har gjort run_step static?
      // enkf_main_run_step(enkf_main , ENSEMBLE_EXPERIMENT , iactive , enopt_start , analyzed , enopt_start , enopt_stop , load_results, false );
      free(iactive);
    }


    enkf_main_free(enkf_main);
    
  }
}
