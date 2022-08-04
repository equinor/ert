enkf_state_type *enkf_main_iget_state(const enkf_main_type *enkf_main,
                                      int iens) {
    return enkf_main->ensemble.at(iens).get();
}
