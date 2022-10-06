#include <stdlib.h>

#include <ert/config/config_parser.hpp>

#include <ert/ecl/ecl_sum.h>

#include <ert/enkf/ensemble_config.hpp>

int main(int argc, char **argv) {
    ensemble_config_type *ensemble = ensemble_config_alloc(NULL, NULL, NULL);
    ensemble_config_free(ensemble);
    exit(0);
}
