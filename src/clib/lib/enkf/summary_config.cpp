#include <stdlib.h>

#include <set>
#include <string>

#include <ert/util/util.h>

#include <ert/ecl/ecl_sum.h>

#include <ert/enkf/summary_config.hpp>

struct summary_config_struct {
    /** The type of the variable - according to ecl_summary nomenclature. */
    ecl_smspec_var_type var_type;
    /** This is ONE variable of summary.x format - i.e. WOPR:OP_2, RPR:4, ... */
    char *var;
};

summary_config_type *summary_config_alloc(const char *var) {
    summary_config_type *config = new summary_config_type();

    config->var = util_alloc_string_copy(var);
    config->var_type = ecl_smspec_identify_var_type(var);

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
