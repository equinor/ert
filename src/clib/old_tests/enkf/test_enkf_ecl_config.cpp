#include <stdlib.h>

#include <ert/util/test_util.h>

#include <ert/enkf/ecl_config.hpp>

int main(int argc, char **argv) {
    ecl_config_type *ecl_config = ecl_config_alloc(NULL);

    if (argc == 2) {
        test_assert_true(ecl_config_load_refcase(ecl_config, argv[1]));

        {
            const ecl_sum_type *def = ecl_config_get_refcase(ecl_config);

            test_assert_string_equal(argv[1], ecl_sum_get_case(def));
        }
    }
    test_assert_false(ecl_config_load_refcase(ecl_config, "DOES_NOT_EXIST"));
    test_assert_true(ecl_config_load_refcase(ecl_config, NULL));

    ecl_config_free(ecl_config);
    exit(0);
}
