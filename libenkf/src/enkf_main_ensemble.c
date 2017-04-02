static void enkf_main_init_jobname( enkf_main_type * enkf_main) {
  for (int iens = 0; iens < enkf_main->ens_size; iens++)
    enkf_state_update_jobname( enkf_main->ensemble[iens] );
}



static void enkf_main_free_ensemble( enkf_main_type * enkf_main ) {
  if (enkf_main->ensemble != NULL) {
    const int ens_size = enkf_main->ens_size;
    int i;
    for (i=0; i < ens_size; i++)
      enkf_state_free( enkf_main->ensemble[i] );
    free(enkf_main->ensemble);
    enkf_main->ensemble = NULL;
  }
}


enkf_state_type * enkf_main_iget_state(const enkf_main_type * enkf_main , int iens) {
  return enkf_main->ensemble[iens];
}




/**
   This function will resize the enkf_main->ensemble vector,
   allocating or freeing enkf_state instances as needed.
*/


void enkf_main_resize_ensemble( enkf_main_type * enkf_main , int new_ens_size ) {
  int iens;

  /* No change */
  if (new_ens_size == enkf_main->ens_size)
    return ;

  ranking_table_set_ens_size( enkf_main->ranking_table , new_ens_size );
  /* Tell the site_config object (i.e. the queue drivers) about the new ensemble size: */
  site_config_set_ens_size( enkf_main->site_config , new_ens_size );


  /* The ensemble is shrinking. */
  if (new_ens_size < enkf_main->ens_size) {
    /*1: Free all ensemble members which go out of scope. */
    for (iens = new_ens_size; iens < enkf_main->ens_size; iens++)
      enkf_state_free( enkf_main->ensemble[iens] );

    /*2: Shrink the ensemble pointer. */
    enkf_main->ensemble = util_realloc(enkf_main->ensemble , new_ens_size * sizeof * enkf_main->ensemble );
    enkf_main->ens_size = new_ens_size;
    return;
  }


  /* The ensemble is expanding */
  if (new_ens_size > enkf_main->ens_size) {
    /*1: Grow the ensemble pointer. */
    enkf_main->ensemble = util_realloc(enkf_main->ensemble , new_ens_size * sizeof * enkf_main->ensemble );

    /*2: Allocate the new ensemble members. */
    for (iens = enkf_main->ens_size; iens < new_ens_size; iens++)

      /* Observe that due to the initialization of the rng - this function is currently NOT thread safe. */
      enkf_main->ensemble[iens] = enkf_state_alloc(iens,
                                                   enkf_main->rng ,
                                                   model_config_iget_casename( enkf_main->model_config , iens ) ,
                                                   enkf_main->pre_clear_runpath                                 ,
                                                   int_vector_safe_iget( enkf_main->keep_runpath , iens)        ,
                                                   enkf_main->model_config                                      ,
                                                   enkf_main->ensemble_config                                   ,
                                                   enkf_main->site_config                                       ,
                                                   enkf_main->ecl_config                                        ,
                                                   enkf_main->templates                                         ,
                                                   enkf_main->subst_list);
    enkf_main->ens_size = new_ens_size;
    return;
  }

  util_abort("%s: something is seriously broken - should NOT be here .. \n",__func__);
}




static void enkf_main_analysis_update( enkf_main_type * enkf_main ,
                                       enkf_fs_type * target_fs ,
                                       const bool_vector_type * ens_mask ,
                                       int target_step ,
                                       hash_type * use_count,
                                       run_mode_type run_mode ,
                                       int step1 ,
                                       int step2 ,
                                       const local_ministep_type * ministep ,
                                       const meas_data_type * forecast ,
                                       obs_data_type * obs_data) {

  const int cpu_threads       = 4;
  const int matrix_start_size = 250000;
  thread_pool_type * tp       = thread_pool_alloc( cpu_threads , false );
  int active_ens_size   = meas_data_get_active_ens_size( forecast );
  int active_size       = obs_data_get_active_size( obs_data );
  matrix_type * X       = matrix_alloc( active_ens_size , active_ens_size );
  matrix_type * S       = meas_data_allocS( forecast );
  matrix_type * R       = obs_data_allocR( obs_data );
  matrix_type * dObs    = obs_data_allocdObs( obs_data );
  matrix_type * A       = matrix_alloc( matrix_start_size , active_ens_size );
  matrix_type * E       = NULL;
  matrix_type * D       = NULL;
  matrix_type * localA  = NULL;
  int_vector_type * iens_active_index = bool_vector_alloc_active_index_list(ens_mask , -1);

  analysis_module_type * module = analysis_config_get_active_module( enkf_main->analysis_config );
  if ( local_ministep_has_analysis_module (ministep))
    module = local_ministep_get_analysis_module (ministep);

  assert_matrix_size(X , "X" , active_ens_size , active_ens_size);
  assert_matrix_size(S , "S" , active_size , active_ens_size);
  assert_matrix_size(R , "R" , active_size , active_size);
  assert_size_equal( enkf_main_get_ensemble_size( enkf_main ) , ens_mask );

  if (analysis_module_check_option( module , ANALYSIS_NEED_ED)) {
    E = obs_data_allocE( obs_data , enkf_main->rng , active_ens_size );
    D = obs_data_allocD( obs_data , E , S );

    assert_matrix_size( E , "E" , active_size , active_ens_size);
    assert_matrix_size( D , "D" , active_size , active_ens_size);
  }

  if (analysis_module_check_option( module , ANALYSIS_SCALE_DATA))
    obs_data_scale( obs_data , S , E , D , R , dObs );

  if (analysis_module_check_option( module , ANALYSIS_USE_A) || analysis_module_check_option(module , ANALYSIS_UPDATE_A))
    localA = A;

  /*****************************************************************/

  analysis_module_init_update( module , ens_mask , S , R , dObs , E , D );
  {
    hash_iter_type * dataset_iter = local_ministep_alloc_dataset_iter( ministep );
    serialize_info_type * serialize_info = serialize_info_alloc( target_fs, //src_fs - we have already copied the parameters from the src_fs to the target_fs
                                                                 target_fs ,
                                                                 iens_active_index,
                                                                 target_step ,
                                                                 enkf_main->ensemble, 
                                                                 run_mode ,
                                                                 step2 ,
                                                                 A ,
                                                                 cpu_threads);


    // Store PC:
    if (analysis_config_get_store_PC( enkf_main->analysis_config )) {
      double truncation    = -1;
      int ncomp            = active_ens_size - 1;
      matrix_type * PC     = matrix_alloc(1,1);
      matrix_type * PC_obs = matrix_alloc(1,1);
      double_vector_type   * singular_values = double_vector_alloc(0,0);
      local_obsdata_type   * obsdata = local_ministep_get_obsdata( ministep );
      const char * obsdata_name = local_obsdata_get_name( obsdata );

      enkf_main_get_PC( S , dObs , truncation , ncomp , PC , PC_obs , singular_values);
      {
        char * filename  = util_alloc_sprintf(analysis_config_get_PC_filename( enkf_main->analysis_config ) , step1 , step2 , obsdata_name);
        char * full_path = util_alloc_filename( analysis_config_get_PC_path( enkf_main->analysis_config) , filename , NULL );

        enkf_main_fprintf_PC( full_path , PC , PC_obs);

        free( full_path );
        free( filename );
      }
      matrix_free( PC );
      matrix_free( PC_obs );
      double_vector_free( singular_values );
    }

    if (localA == NULL)
      analysis_module_initX( module , X , NULL , S , R , dObs , E , D );


    while (!hash_iter_is_complete( dataset_iter )) {
      const char * dataset_name = hash_iter_get_next_key( dataset_iter );
      const local_dataset_type * dataset = local_ministep_get_dataset( ministep , dataset_name );
      if (local_dataset_get_size( dataset )) {
        int * active_size = util_calloc( local_dataset_get_size( dataset ) , sizeof * active_size );
        int * row_offset  = util_calloc( local_dataset_get_size( dataset ) , sizeof * row_offset  );
        local_obsdata_type   * local_obsdata = local_ministep_get_obsdata( ministep );

        enkf_main_serialize_dataset( enkf_main->ensemble_config , dataset , step2 ,  use_count , active_size , row_offset , tp , serialize_info);
        module_info_type * module_info = enkf_main_module_info_alloc(ministep, obs_data, dataset, local_obsdata, active_size , row_offset);

        if (analysis_module_check_option( module , ANALYSIS_UPDATE_A)){
          if (analysis_module_check_option( module , ANALYSIS_ITERABLE)){
            analysis_module_updateA( module , localA , S , R , dObs , E , D , module_info );
          }
          else
            analysis_module_updateA( module , localA , S , R , dObs , E , D , module_info );
        }
        else {
          if (analysis_module_check_option( module , ANALYSIS_USE_A)){
            analysis_module_initX( module , X , localA , S , R , dObs , E , D );
          }

          matrix_inplace_matmul_mt2( A , X , tp );
        }

        // The deserialize also calls enkf_node_store() functions.
        enkf_main_deserialize_dataset( enkf_main_get_ensemble_config( enkf_main ) , dataset , active_size , row_offset , serialize_info , tp);

        free( active_size );
        free( row_offset );
        enkf_main_module_info_free( module_info );
      }
    }
    hash_iter_free( dataset_iter );
    serialize_info_free( serialize_info );
  }
  analysis_module_complete_update( module );


  /*****************************************************************/

  int_vector_free(iens_active_index);
  matrix_safe_free( E );
  matrix_safe_free( D );
  matrix_free( S );
  matrix_free( R );
  matrix_free( dObs );
  matrix_free( X );
  matrix_free( A );
}
