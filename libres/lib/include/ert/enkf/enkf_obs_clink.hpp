extern "C" {

void enkf_obs_free(enkf_obs_type *enkf_obs);
int enkf_obs_get_size(const enkf_obs_type *obs);
bool enkf_obs_is_valid(const enkf_obs_type *);
void enkf_obs_clear(enkf_obs_type *enkf_obs);
stringlist_type *enkf_obs_alloc_typed_keylist(enkf_obs_type *enkf_obs,
                                              obs_impl_type);
stringlist_type *enkf_obs_alloc_matching_keylist(const enkf_obs_type *enkf_obs,
                                                 const char *input_string);
bool enkf_obs_has_key(const enkf_obs_type *, const char *);
obs_impl_type enkf_obs_get_type(const enkf_obs_type *enkf_obs, const char *key);
obs_vector_type *enkf_obs_get_vector(const enkf_obs_type *, const char *);
obs_vector_type *enkf_obs_iget_vector(const enkf_obs_type *obs, int index);
time_t enkf_obs_iget_obs_time(const enkf_obs_type *enkf_obs, int report_step);
void enkf_obs_add_obs_vector(enkf_obs_type *enkf_obs,
                             const obs_vector_type *vector);
}
