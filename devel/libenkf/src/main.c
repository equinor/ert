#include <enkf_fs.h>
#include <enkf_main.h>
#include <util.h>
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
#include <enkf_ui_main.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>



void install_SIGNALS(void) {
  signal(SIGSEGV , util_abort_signal);
}


void text_splash() {
  const int usleep_time = 2500;
  int i;
  {
#include "uncle_sam_100.inc"
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
  text_splash();
  enkf_welcome();
  install_SIGNALS();

  if (argc != 2) {
    enkf_usage();
    exit(1);
  } else {
    const char * site_config_file  = SITE_CONFIG_FILE;  /* The variable SITE_CONFIG_FILE should be defined on compilation ... */
    const char * model_config_file = argv[1]; 

    enkf_main_type * enkf_main = enkf_main_bootstrap(site_config_file , model_config_file);
    enkf_ui_main_menu(enkf_main); 
    enkf_main_free(enkf_main);
    
  }
}
