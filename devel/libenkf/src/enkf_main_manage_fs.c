/*
   Copyright (C) 2013  Statoil ASA, Norway.
    
   The file 'enkf_main_manage_fs.c' is part of ERT - Ensemble based Reservoir Tool.
    
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

#include <ert/enkf/enkf_main.h>

static void enkf_main_copy_ensemble( const enkf_main_type * enkf_main,
                                     enkf_fs_type * source_case_fs,
                                     int source_report_step,
                                     state_enum source_state,
                                     enkf_fs_type * target_case_fs,
                                     int target_report_step,
                                     state_enum target_state,
                                     const bool_vector_type * iens_mask,
                                     const char * ranking_key , /* It is OK to supply NULL - but if != NULL it must exist */
                                     const stringlist_type * node_list) {

  const int ens_size = enkf_main_get_ensemble_size( enkf_main );

  {
    int * ranking_permutation;
    int inode , src_iens;

    if (ranking_key != NULL) {
      ranking_table_type * ranking_table = enkf_main_get_ranking_table( enkf_main );
      ranking_permutation = (int *) ranking_table_get_permutation( ranking_table , ranking_key );
    } else {
      ranking_permutation = util_calloc( ens_size , sizeof * ranking_permutation );
      for (src_iens = 0; src_iens < ens_size; src_iens++)
        ranking_permutation[src_iens] = src_iens;
    }

    for (inode =0; inode < stringlist_get_size( node_list ); inode++) {
      enkf_config_node_type * config_node = ensemble_config_get_node( enkf_main_get_ensemble_config(enkf_main) , stringlist_iget( node_list , inode ));
      for (src_iens = 0; src_iens < enkf_main_get_ensemble_size( enkf_main ); src_iens++) {
        if (bool_vector_safe_iget(iens_mask , src_iens)) {
          int target_iens = ranking_permutation[src_iens];
          node_id_type src_id    = {.report_step = source_report_step , .iens = src_iens    , .state = source_state };
          node_id_type target_id = {.report_step = target_report_step , .iens = target_iens , .state = target_state };

          printf("CALLING enkf_node_copy!!!\n");
          enkf_node_copy( config_node ,
                          source_case_fs , target_case_fs ,
                          src_id , target_id );
        }
      }
    }

    if (ranking_permutation == NULL)
      free( ranking_permutation );
  }
}


bool enkf_main_case_is_current(const enkf_main_type * enkf_main , const char * case_path) {
  char * mount_point               = enkf_main_alloc_mount_point( enkf_main , case_path );
  const char * current_mount_point = NULL;
  bool is_current;

  if (enkf_main->dbase != NULL)
    current_mount_point = enkf_fs_get_mount_point( enkf_main->dbase );

  is_current = util_string_equal( mount_point , current_mount_point );
  free( mount_point );
  return is_current;
}


char* enkf_main_read_alloc_current_case_name(const enkf_main_type * enkf_main) {
  char * current_case = NULL;
  const char * ens_path = model_config_get_enspath( enkf_main->model_config);
  char * current_case_file = util_alloc_filename(ens_path, CURRENT_CASE_FILE, NULL);
  if (enkf_main_current_case_file_exists(enkf_main)) {
    FILE * stream = util_fopen( current_case_file  , "r");
    current_case = util_fscanf_alloc_token(stream);
    util_fclose(stream);
  } else {
    util_abort("%s: File: storage/current_case not found, aborting! \n",__func__);
  }
  free(current_case_file);
  return current_case;
}


stringlist_type * enkf_main_alloc_caselist( const enkf_main_type * enkf_main ) {
  stringlist_type * case_list = stringlist_alloc_new( );
  {
    const char * ens_path = model_config_get_enspath( enkf_main->model_config );
    DIR * ens_dir = opendir( ens_path );
    if (ens_dir != NULL) {
      int ens_fd = dirfd( ens_dir );
      if (ens_fd != -1) {
        struct dirent * dp;
        do {
          dp = readdir( ens_dir );
          if (dp != NULL) {
            if (!(util_string_equal( dp->d_name , ".") || util_string_equal(dp->d_name , ".."))) {
              if (!util_string_equal( dp->d_name , CURRENT_CASE_FILE)) {
                char * full_path = util_alloc_filename( ens_path , dp->d_name , NULL);
                if (util_is_directory( full_path ))
                  stringlist_append_copy( case_list , dp->d_name );
                free( full_path);
              }
            }
          }
        } while (dp != NULL);
      }
    }
    closedir( ens_dir );
  }
  return case_list;
}

void enkf_main_init_current_case_from_existing(const enkf_main_type * enkf_main,
                                               enkf_fs_type * source_case_fs,
                                               int source_report_step,
                                               state_enum source_state) {

  enkf_fs_type * current_fs = enkf_main_get_fs(enkf_main);

  enkf_main_init_case_from_existing(enkf_main,
                                    source_case_fs,
                                    source_report_step,
                                    source_state,
                                    current_fs);
}


void enkf_main_init_case_from_existing(const enkf_main_type * enkf_main,
                                       enkf_fs_type * source_case_fs,
                                       int source_report_step,
                                       state_enum source_state,
                                       enkf_fs_type * target_case_fs ) {

  stringlist_type * param_list = ensemble_config_alloc_keylist_from_var_type( enkf_main_get_ensemble_config(enkf_main) , PARAMETER ); /* Select only paramters - will fail for GEN_DATA of type DYNAMIC_STATE. */
  int target_report_step  = 0;
  state_enum target_state = ANALYZED;
  bool_vector_type * iactive = bool_vector_alloc( 0 , true );

  enkf_main_copy_ensemble(enkf_main,
                          source_case_fs,
                          source_report_step,
                          source_state,
                          target_case_fs,
                          target_report_step,
                          target_state ,
                          iactive,
                          NULL,
                          param_list);

  bool_vector_free(iactive);
  stringlist_free(param_list);
  printf("FINISHED!\n");
}


bool enkf_main_case_is_initialized( const enkf_main_type * enkf_main , const char * case_name ,  bool_vector_type * __mask) {
  enkf_fs_type * fs = enkf_main_mount_alt_fs(enkf_main , case_name , true , false);
  if (fs) {
    bool initialized = enkf_main_case_is_initialized__(enkf_main , fs , __mask);
    enkf_fs_umount( fs );
    return initialized;
  } else
    return false;
}


bool enkf_main_set_refcase( enkf_main_type * enkf_main , const char * refcase_path) {
  bool set_refcase = ecl_config_load_refcase( enkf_main->ecl_config , refcase_path );

  model_config_set_refcase( enkf_main->model_config , ecl_config_get_refcase( enkf_main->ecl_config ));
  ensemble_config_set_refcase( enkf_main->ensemble_config , ecl_config_get_refcase( enkf_main->ecl_config ));

  return set_refcase;
}


ui_return_type * enkf_main_validata_refcase( const enkf_main_type * enkf_main , const char * refcase_path) {
  return ecl_config_validate_refcase( enkf_main->ecl_config , refcase_path );
}


void enkf_main_set_case_table( enkf_main_type * enkf_main , const char * case_table_file ) {
  model_config_set_case_table( enkf_main->model_config , enkf_main->ens_size , case_table_file );
}

