/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'enkf_main_jobs.c' is part of ERT - Ensemble based Reservoir Tool. 
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
*/

#include <ert/util/stringlist.h>
#include <ert/util/string_util.h>
#include <ert/util/int_vector.h>

#include <ert/enkf/enkf_main.h>
#include <ert/enkf/field_config.h>


void * enkf_main_exit_JOB(void * self , const stringlist_type * args ) {
  enkf_main_type  * enkf_main = enkf_main_safe_cast( self );
  enkf_main_exit( enkf_main );
  return NULL;
}


void * enkf_main_assimilation_JOB( void * self , const stringlist_type * args ) {
  enkf_main_type   * enkf_main = enkf_main_safe_cast( self );
  int ens_size                 = enkf_main_get_ensemble_size( enkf_main );
  bool_vector_type * iactive   = bool_vector_alloc( 0 , true );

  bool_vector_iset( iactive , ens_size - 1 , true );
  enkf_main_run_assimilation( enkf_main , iactive , 0 , 0 ,  ANALYZED );
  return NULL;
}


void * enkf_main_ensemble_run_JOB( void * self , const stringlist_type * args ) {
  enkf_main_type   * enkf_main = enkf_main_safe_cast( self );
  int ens_size                 = enkf_main_get_ensemble_size( enkf_main );
  bool_vector_type * iactive   = bool_vector_alloc( 0 , true );

  // Ignore args until string_utils is in place ..... 
  // if (stringlist_get_size( args ) 

  bool_vector_iset( iactive , ens_size - 1 , true );
  enkf_main_run_exp( enkf_main , iactive , true , 0 , 0 , ANALYZED);
  return NULL;
}


void * enkf_main_smoother_JOB( void * self , const stringlist_type * args ) {
  enkf_main_type   * enkf_main = enkf_main_safe_cast( self );
  int ens_size                 = enkf_main_get_ensemble_size( enkf_main );
  bool_vector_type * iactive   = bool_vector_alloc( ens_size , true );
  const char * target_case     = stringlist_iget( args , 0 );
  bool valid                   = true;
  bool rerun = (stringlist_get_size(args) >= 2) ? stringlist_iget_as_bool(args, 1, &valid) : true;

  if (!valid) {
      fprintf(stderr, "** Warning: Function %s : Second argument must be a bool value. Exiting job\n", __func__);
      return NULL;
  }

  enkf_main_run_smoother( enkf_main , target_case , iactive , rerun);
  bool_vector_free( iactive );
  return NULL;
}


void * enkf_main_iterated_smoother_JOB( void * self , const stringlist_type * args ) {
  enkf_main_type   * enkf_main = enkf_main_safe_cast( self );
  int ens_size                 = enkf_main_get_ensemble_size( enkf_main );
  bool_vector_type * iactive   = bool_vector_alloc( 0 , true );

  bool_vector_iset( iactive , ens_size - 1 , true );
  enkf_main_run_iterated_ES( enkf_main);
  return NULL;
}


void * enkf_main_select_module_JOB( void * self , const stringlist_type * args ) {
  enkf_main_type   * enkf_main = enkf_main_safe_cast( self );
  analysis_config_type * analysis_config = enkf_main_get_analysis_config( enkf_main );
  
  analysis_config_select_module( analysis_config , stringlist_iget( args , 0 ));
  
  return NULL;
}



void * enkf_main_create_reports_JOB(void * self , const stringlist_type * args ) {
  enkf_main_type   * enkf_main = enkf_main_safe_cast( self );
  ert_report_list_type * report_list = enkf_main_get_report_list( enkf_main );
  
  ert_report_list_create( report_list , enkf_main_get_current_fs( enkf_main ) , true );
  return NULL;
}

void * enkf_main_scale_obs_std_JOB(void * self, const stringlist_type * args ) {
  enkf_main_type   * enkf_main = enkf_main_safe_cast( self );
  
  double scale_factor;
  util_sscanf_double(stringlist_iget(args, 0), &scale_factor);

  if (enkf_main_have_obs(enkf_main)) {
    enkf_obs_type * observations = enkf_main_get_obs(enkf_main);
    enkf_obs_scale_std(observations, scale_factor);
  }
  return NULL;
}

/* 
   Will create the new case if it does not exist. 
*/
void * enkf_main_select_case_JOB( void * self , const stringlist_type * args) {
  enkf_main_type * enkf_main = enkf_main_safe_cast( self );
  const char * new_case = stringlist_iget( args , 0 );
  enkf_main_select_fs( enkf_main , new_case );
  return NULL;
}


void * enkf_main_create_case_JOB( void * self , const stringlist_type * args) {
  enkf_main_type * enkf_main = enkf_main_safe_cast( self );
  const char * new_case = stringlist_iget( args , 0 );
  enkf_fs_type * fs = enkf_main_mount_alt_fs( enkf_main , new_case , false , true );
  enkf_fs_umount(fs);
  return NULL;
}


void * enkf_main_load_results_JOB( void * self , const stringlist_type * args) {
  enkf_main_type * enkf_main = enkf_main_safe_cast( self );
  bool_vector_type * iactive = alloc_iactive_list(enkf_main_get_ensemble_size(enkf_main), args, 0);
  enkf_main_load_from_forward_model(enkf_main, iactive, NULL);
  bool_vector_free(iactive);
  return NULL;
}


/*****************************************************************/

static void enkf_main_jobs_export_field(const enkf_main_type * enkf_main, const stringlist_type * args, field_file_format_type file_type) {
  const char *      field            = stringlist_iget(args, 0); 
  const char *      file_name        = stringlist_iget(args, 1); 
  int               report_step      = 0;
  util_sscanf_int(stringlist_iget(args,2), &report_step);
  state_enum        state            = enkf_types_get_state_enum(stringlist_iget(args, 3)); 

  if (BOTH == state) {
      fprintf(stderr,"** Field export jobs only supports state_enum ANALYZED or FORECAST, not BOTH.\n");
      return;
  }

  bool_vector_type * iactive = alloc_iactive_list(enkf_main_get_ensemble_size(enkf_main), args, 4);
  enkf_main_export_field(enkf_main,field, file_name, iactive, file_type, report_step, state) ;
  bool_vector_free(iactive);
}



void * enkf_main_export_field_JOB(void * self, const stringlist_type * args) {
  const char * file_name = stringlist_iget(args, 1); 
  field_file_format_type file_type = field_config_default_export_format(file_name); 
  
  if ((RMS_ROFF_FILE == file_type) || (ECL_GRDECL_FILE == file_type)) {
    enkf_main_type * enkf_main = enkf_main_safe_cast( self );
    enkf_main_jobs_export_field(enkf_main, args, file_type);
  } else
    printf("EXPORT_FIELD filename argument: File extension must be either .roff or .grdecl\n"); 
    
  return NULL; 
}

void * enkf_main_export_field_to_RMS_JOB(void * self, const stringlist_type * args) {
  enkf_main_type * enkf_main = enkf_main_safe_cast( self );
  enkf_main_jobs_export_field(enkf_main, args, RMS_ROFF_FILE);
  return NULL; 
}

void * enkf_main_export_field_to_ECL_JOB(void * self, const stringlist_type * args) {
  enkf_main_type * enkf_main = enkf_main_safe_cast( self );
  enkf_main_jobs_export_field(enkf_main, args, ECL_GRDECL_FILE);
  return NULL; 
}






