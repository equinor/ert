extern "C" {

bool enkf_obs_is_valid(const enkf_obs_type *obs) { return obs->valid; }

void enkf_obs_free(enkf_obs_type *enkf_obs) {
    hash_free(enkf_obs->obs_hash);
    vector_free(enkf_obs->obs_vector);
    time_map_free(enkf_obs->obs_time);
    free(enkf_obs);
}

time_t enkf_obs_iget_obs_time(const enkf_obs_type *enkf_obs, int report_step) {
    return time_map_iget(enkf_obs->obs_time, report_step);
}

bool enkf_obs_has_key(const enkf_obs_type *obs, const char *key) {
    return hash_has_key(obs->obs_hash, key);
}

obs_vector_type *enkf_obs_get_vector(const enkf_obs_type *obs,
                                     const char *key) {
    return (obs_vector_type *)hash_get(obs->obs_hash, key);
}

obs_vector_type *enkf_obs_iget_vector(const enkf_obs_type *obs, int index) {
    return (obs_vector_type *)vector_iget(obs->obs_vector, index);
}

int enkf_obs_get_size(const enkf_obs_type *obs) {
    return vector_get_size(obs->obs_vector);
}

void enkf_obs_clear(enkf_obs_type *enkf_obs) {
    hash_clear(enkf_obs->obs_hash);
    vector_clear(enkf_obs->obs_vector);
    ensemble_config_clear_obs_keys(enkf_obs->ensemble_config);
}

/*
      Allocates a stringlist of obs target keys which correspond to
      summary observations, these are then added to the state vector in
      enkf_main.
    */
stringlist_type *enkf_obs_alloc_typed_keylist(enkf_obs_type *enkf_obs,
                                              obs_impl_type obs_type) {
    stringlist_type *vars = stringlist_alloc_new();
    hash_iter_type *iter = hash_iter_alloc(enkf_obs->obs_hash);
    const char *key = hash_iter_get_next_key(iter);
    while (key != NULL) {
        obs_vector_type *obs_vector =
            (obs_vector_type *)hash_get(enkf_obs->obs_hash, key);
        if (obs_vector_get_impl_type(obs_vector) == obs_type)
            stringlist_append_copy(vars, key);
        key = hash_iter_get_next_key(iter);
    }
    hash_iter_free(iter);
    return vars;
}

obs_impl_type enkf_obs_get_type(const enkf_obs_type *enkf_obs,
                                const char *key) {
    obs_vector_type *obs_vector =
        (obs_vector_type *)hash_get(enkf_obs->obs_hash, key);
    return obs_vector_get_impl_type(obs_vector);
}

stringlist_type *enkf_obs_alloc_matching_keylist(const enkf_obs_type *enkf_obs,
                                                 const char *input_string) {

    stringlist_type *obs_keys = hash_alloc_stringlist(enkf_obs->obs_hash);

    if (!input_string)
        return obs_keys;

    stringlist_type *matching_keys = stringlist_alloc_new();
    char **input_keys;
    int num_keys;
    int obs_keys_count = stringlist_get_size(obs_keys);

    util_split_string(input_string, " ", &num_keys, &input_keys);
    for (int i = 0; i < num_keys; i++) {
        const char *input_key = input_keys[i];
        for (int j = 0; j < obs_keys_count; j++) {
            const char *obs_key = stringlist_iget(obs_keys, j);

            if (util_string_match(obs_key, input_key) &&
                !stringlist_contains(matching_keys, obs_key))
                stringlist_append_copy(matching_keys, obs_key);
        }
    }
    util_free_stringlist(input_keys, num_keys);
    stringlist_free(obs_keys);
    return matching_keys;
}

/*
      Observe that the obs_vector can be NULL - in which it is of course not added.
    */
void enkf_obs_add_obs_vector(enkf_obs_type *enkf_obs,
                             const obs_vector_type *vector) {

    if (vector != NULL) {
        const char *obs_key = obs_vector_get_key(vector);
        if (hash_has_key(enkf_obs->obs_hash, obs_key))
            util_abort("%s: Observation with key:%s already added.\n", __func__,
                       obs_key);

        hash_insert_ref(enkf_obs->obs_hash, obs_key, vector);
        vector_append_owned_ref(enkf_obs->obs_vector, vector,
                                obs_vector_free__);
    }
}
}
