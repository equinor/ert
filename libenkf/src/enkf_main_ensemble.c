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
                                                   enkf_main->rng,
                                                   model_config_iget_casename(enkf_main->model_config, iens),
                                                   enkf_main->pre_clear_runpath,
                                                   int_vector_safe_iget(enkf_main->keep_runpath, iens),
                                                   enkf_main->model_config,
                                                   enkf_main_get_ensemble_config(enkf_main),
                                                   enkf_main_get_site_config(enkf_main),
                                                   enkf_main_get_ecl_config(enkf_main),
                                                   enkf_main_get_templates(enkf_main),
                                                   enkf_main_get_data_kw(enkf_main));
    enkf_main->ens_size = new_ens_size;
    return;
  }

  util_abort("%s: something is seriously broken - should NOT be here .. \n",__func__);
}




