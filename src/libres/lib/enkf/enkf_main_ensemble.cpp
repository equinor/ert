

static void enkf_main_free_ensemble(enkf_main_type *enkf_main) {
    if (enkf_main->ensemble != NULL) {
        const int ens_size = enkf_main->ens_size;
        int i;
        for (i = 0; i < ens_size; i++)
            enkf_state_free(enkf_main->ensemble[i]);
        free(enkf_main->ensemble);
        enkf_main->ensemble = NULL;
    }
}

enkf_state_type *enkf_main_iget_state(const enkf_main_type *enkf_main,
                                      int iens) {
    return enkf_main->ensemble[iens];
}

void enkf_main_increase_ensemble(enkf_main_type *enkf_main, int new_ens_size) {
    int iens;

    /* No change */
    if (new_ens_size == enkf_main->ens_size)
        return;

    /* The ensemble is expanding */
    if (new_ens_size > enkf_main->ens_size) {
        /*1: Grow the ensemble pointer. */
        enkf_main->ensemble = (enkf_state_type **)util_realloc(
            enkf_main->ensemble, new_ens_size * sizeof *enkf_main->ensemble);

        /*2: Allocate the new ensemble members. */
        for (iens = enkf_main->ens_size; iens < new_ens_size; iens++)

            /* Observe that due to the initialization of the rng - this function is currently NOT thread safe. */
            enkf_main->ensemble[iens] = enkf_state_alloc(
                iens, rng_manager_iget(enkf_main->rng_manager, iens),
                enkf_main_get_model_config(enkf_main),
                enkf_main_get_ensemble_config(enkf_main),
                enkf_main_get_site_config(enkf_main),
                enkf_main_get_ecl_config(enkf_main),
                res_config_get_templates(enkf_main->res_config));
        enkf_main->ens_size = new_ens_size;
        return;
    }

    util_abort("%s: something is seriously broken - should NOT be here .. \n",
               __func__);
}
