#include <stdlib.h>

#include <set>
#include <string>

#include <ert/util/util.h>

#include <ert/ecl/ecl_sum.h>

#include <ert/enkf/summary_config.hpp>

struct summary_config_struct {
    load_fail_type load_fail;
    /** The type of the variable - according to ecl_summary nomenclature. */
    ecl_smspec_var_type var_type;
    /** This is ONE variable of summary.x format - i.e. WOPR:OP_2, RPR:4, ... */
    char *var;
    /** Set of keys (which fit in enkf_obs) which are observations of this node. */
    std::set<std::string> obs_set;
};

const char *summary_config_get_var(const summary_config_type *config) {
    return config->var;
}

load_fail_type
summary_config_get_load_fail_mode(const summary_config_type *config) {
    return config->load_fail;
}

/**
   Unfortunately it is a bit problematic to set the required flag to
   TRUE for well and group variables because they do not exist in the
   summary results before the well has actually opened, i.e. for a
   partial summary case the results will not be there, and the loader
   will incorrectly(?) signal failure.
*/
void summary_config_set_load_fail_mode(summary_config_type *config,
                                       load_fail_type load_fail) {
    if ((config->var_type == ECL_SMSPEC_WELL_VAR) ||
        (config->var_type == ECL_SMSPEC_GROUP_VAR))
        // For well and group variables load_fail will be LOAD_FAIL_SILENT anyway.
        config->load_fail = LOAD_FAIL_SILENT;
    else
        config->load_fail = load_fail;
}

/**
   This can only be used to increase the load_fail strictness.
*/
void summary_config_update_load_fail_mode(summary_config_type *config,
                                          load_fail_type load_fail) {
    if (load_fail > config->load_fail)
        summary_config_set_load_fail_mode(config, load_fail);
}

summary_config_type *summary_config_alloc(const char *var,
                                          load_fail_type load_fail) {
    summary_config_type *config = new summary_config_type();

    config->var = util_alloc_string_copy(var);
    config->var_type = ecl_smspec_identify_var_type(var);
    summary_config_set_load_fail_mode(config, load_fail);

    return config;
}

void summary_config_free(summary_config_type *config) {
    free(config->var);
    delete config;
}

int summary_config_get_data_size(const summary_config_type *config) {
    return 1;
}

VOID_GET_DATA_SIZE(summary)
VOID_CONFIG_FREE(summary)
