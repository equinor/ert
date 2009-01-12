#include <sys/types.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <util.h>
#include <stdlib.h>
#include <string.h>
#include <enkf_fs.h>
#include <path_fmt.h>
#include <enkf_sched.h>
#include <model_config.h>
#include <hash.h>
#include <history.h>
#include <config.h>
#include <sched_file.h>
#include <ecl_sum.h>
#include <ecl_util.h>
#include <ecl_grid.h>
#include <menu.h>
#include <enkf_types.h>
#include <plain_driver.h>

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
  stringlist_type   * forward_model;      /* A list of external jobs - which acts as keys into a ext_joblist_type instance. */
  path_fmt_type     * result_path;        /* path_fmt instance for results - should contain one %d which will be replaced report_step */
  path_fmt_type     * runpath;            /* path_fmt instance for runpath - runtime the call gets arguments: (iens, report_step1 , report_step2) - i.e. at least one %d must be present.*/  
  enkf_sched_type   * enkf_sched;         /* The enkf_sched object controlling when the enkf is ON|OFF, strides in report steps and special forward model. */
  char              * lock_path;          /* Path containing lock files */
  lock_mode_type      runlock_mode;       /* Mode for locking run directories - currently not working.*/ 
};



static enkf_fs_type * fs_mount(const char * root_path , const char * lock_path) {
  const char * mount_map = "enkf_mount_info";
  char * config_file     = util_alloc_full_path(root_path , mount_map); /* This file should be protected - at all costs. */
  
  util_make_path(root_path);
  util_unlink_existing( config_file ); /* Format of mount map changed aby adding obs driver at revision 1505 - MUST read this file. */
  
  if ( !util_file_exists(config_file)) {  
    enkf_fs_fwrite_new_mount_map( config_file , "enkf" );
    //int fd        = open(config_file , O_WRONLY + O_CREAT);
    //FILE * stream = fdopen(fd, "w");
    //
    //plain_driver_fwrite_mount_info( stream );
    ///* 
    //   Changing mode to read-only in an attempt to protect the file.
    //   A better solution would be to create the file in a
    //   write-protected directory.
    //*/
    //fchmod(fd , S_IRUSR + S_IRGRP + S_IROTH); 
    //fclose(stream);
    //close(fd);
  }
  free(config_file);
  return enkf_fs_mount(root_path , mount_map , lock_path);
}


void model_config_set_runpath_fmt(model_config_type * model_config, const char * fmt){
  if (model_config->runpath != NULL)
    path_fmt_free( model_config->runpath );
  
  model_config->runpath  = path_fmt_alloc_directory_fmt( fmt );
}



model_config_type * model_config_alloc(const config_type * config , const ext_joblist_type * joblist , const sched_file_type * sched_file) {
  int num_restart_files = sched_file_get_num_restart_files(sched_file);
  model_config_type * model_config = util_malloc(sizeof * model_config , __func__);

  model_config->result_path    = path_fmt_alloc_directory_fmt( config_get(config , "RESULT_PATH") );
  model_config->forward_model  = config_alloc_stringlist( config , "FORWARD_MODEL" );
  model_config->enkf_sched     = enkf_sched_fscanf_alloc( config_safe_get(config , "ENKF_SCHED_FILE") , num_restart_files  , joblist , model_config->forward_model);
  model_config->runlock_mode   = lock_none;
  {
    char * cwd = util_alloc_cwd();
    model_config->lock_path      = util_alloc_full_path(cwd , "locks");
    free(cwd);
  }
  util_make_path( model_config->lock_path );
  model_config->ensemble_dbase = fs_mount( config_get(config , "ENSPATH") , model_config->lock_path);
  model_config->runpath = NULL;
  model_config_set_runpath_fmt( model_config , config_get(config , "RUNPATH") );
  model_config->history = history_alloc_from_sched_file(sched_file);  
  {
    const char * history_source = config_get(config , "HISTORY_SOURCE");
    const char * refcase        = NULL;
    bool  use_history;

    if (strcmp(history_source , "REFCASE_OBSERVED") == 0) {
      refcase = config_get(config , "REFCASE");
      use_history = false;
    } else if (strcmp(history_source , "REFCASE_HISTORY") == 0) {
      refcase = config_get(config , "REFCASE");
      use_history = true;
    }

    if ((refcase != NULL) && (strcmp(history_source , "SCHEDULE") != 0)) {
      char  * refcase_path;
      char  * refcase_base;
      char  * header_file;
      char ** summary_file_list;
      int     files;
      bool    fmt_file ,unified;
      ecl_sum_type * ecl_sum;
      
      util_alloc_file_components( refcase , &refcase_path , &refcase_base , NULL);
      printf("Loading summary from: %s \n",refcase_path);
      ecl_util_alloc_summary_files( refcase_path , refcase_base , &header_file , &summary_file_list , &files , &fmt_file , &unified);

      ecl_sum = ecl_sum_fread_alloc( header_file , files , (const char **) summary_file_list , true , true /* Endian convert */);
      history_realloc_from_summary( model_config->history , ecl_sum , use_history);        
      util_safe_free(header_file);
      util_safe_free(refcase_base);
      util_safe_free(refcase_path);
      util_free_stringlist(summary_file_list, files);
      ecl_sum_free(ecl_sum);
    }
  }

  
  return model_config;
}




void model_config_free(model_config_type * model_config) {
  path_fmt_free(  model_config->result_path );
  path_fmt_free(  model_config->runpath );
  enkf_sched_free( model_config->enkf_sched );
  free(model_config->lock_path);
  enkf_fs_free(model_config->ensemble_dbase);
  history_free(model_config->history);
  stringlist_free(model_config->forward_model);
  free(model_config);
}


enkf_fs_type * model_config_get_fs(const model_config_type * model_config) {
  return model_config->ensemble_dbase;
}


path_fmt_type * model_config_get_runpath_fmt(const model_config_type * model_config) {
  return model_config->runpath;
}

char * model_config_alloc_result_path(const model_config_type * config , int report_step) {
  return path_fmt_alloc_path(config->result_path , true , report_step);
}

enkf_sched_type * model_config_get_enkf_sched(const model_config_type * config) {
  return config->enkf_sched;
}

history_type * model_config_get_history(const model_config_type * config) {
  return config->history;
}


void model_config_interactive_set_runpath__(void * arg) {
  arg_pack_type * arg_pack = arg_pack_safe_cast( arg );
  model_config_type * model_config = arg_pack_iget_ptr(arg_pack , 0);
  menu_item_type    * item         = arg_pack_iget_ptr(arg_pack , 1);
  char runpath_fmt[256];
  printf("Give runpath format ==> ");
  scanf("%s" , runpath_fmt);
  model_config_set_runpath_fmt(model_config , runpath_fmt);
  {
    char * menu_label = util_alloc_sprintf("Set new value for RUNPATH:%s" , runpath_fmt);
    menu_item_set_label( item , menu_label );
    free(menu_label);
  }
}
