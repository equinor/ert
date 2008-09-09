#include <enkf_fs.h>
#include <enkf_main.h>
#include <enkf_config.h>
#include <enkf_site_config.h>
#include <util.h>
#include <basic_queue_driver.h>
#include <plain_driver_dynamic.h>
#include <plain_driver_parameter.h>
#include <plain_driver_static.h>
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
  int i;
  {
#include "statoilhydro.inc"
    printf("\n\n");
    for (i = 0; i < SPLASH_LENGTH; i++)
      printf("%s\n" , splash_text[i]);
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



enkf_fs_type * fs_mount(const char * root_path) {
  const char * mount_map = "enkf_mount_info";
  const char * config_file = util_alloc_full_path(root_path , mount_map); /* This file should be protected - at all costs. */
  
  util_make_path(root_path);
  if (!util_file_exists(config_file)) {
    int fd        = open(config_file , O_WRONLY + O_CREAT);
    FILE * stream = fdopen(fd, "w");

    plain_driver_parameter_fwrite_mount_info(stream , "%04d/mem%03d/Parameter"); 
    plain_driver_static_fwrite_mount_info(stream    , "%04d/mem%03d/Static"); 
    plain_driver_dynamic_fwrite_mount_info(stream   , "%04d/mem%03d/Forecast", "%04d/mem%03d/Analyzed");
    fs_index_fwrite_mount_info(stream , "%04d/mem%03d/INDEX");

    /* 
       Changing mode to read-only in an attempt to protect the file.
       A better solution would be to create the file in a
       write-protected directory.
    */
    fchmod(fd , S_IRUSR + S_IRGRP + S_IROTH); 
    fclose(stream);
    close(fd);
  }
  
  return enkf_fs_mount(root_path , mount_map);
}


int main (int argc , char ** argv) {
  text_splash();
  enkf_welcome();
  install_SIGNALS();

  if (argc != 2) {
    enkf_usage();
    exit(1);
  } else {
    lock_mode_type lock_mode = lock_file;
    const char * site_config_file = SITE_CONFIG_FILE;  /* The variable SITE_CONFIG_FILE should be defined on compilation ... */
    const char * config_file      = argv[1];
    ext_joblist_type * joblist;
    job_queue_type   * job_queue;
    enkf_main_type   * enkf_main;
    enkf_site_config_type * site_config = enkf_site_config_bootstrap(site_config_file);
    joblist   = ext_joblist_alloc();

    enkf_config_type  * enkf_config              = enkf_config_fscanf_alloc(config_file , site_config , joblist , false , false , true);
    enkf_fs_type      * fs = fs_mount( enkf_config_get_ens_path(enkf_config) );
    char * lock_path = util_alloc_full_path (getenv("CWD") , "locks");
    util_make_path(lock_path);

    job_queue = enkf_config_alloc_job_queue(enkf_config , site_config);
    enkf_main = enkf_main_alloc(enkf_config , lock_mode , lock_path , fs , job_queue , joblist);
    const enkf_sched_type * enkf_sched = enkf_sched_fscanf_alloc( enkf_config_get_enkf_sched_file(enkf_config) , enkf_main_get_sched_file(enkf_main) , joblist , enkf_config_get_forward_model(enkf_config));
    
    enkf_ui_main_menu(enkf_main , enkf_sched);
        
    job_queue_free(job_queue);
    enkf_main_free(enkf_main);
    enkf_site_config_free(site_config); /* Should probably be owned by enkf_main ?? */

    ext_joblist_free(joblist);
    enkf_fs_free(fs);  /* Takes the drivers as well */
  }
}
