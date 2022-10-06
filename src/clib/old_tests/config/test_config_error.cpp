#include <stdlib.h>

#include <ert/util/test_util.hpp>

#include <ert/config/config_error.hpp>

int main(int argc, char **argv) {
    config_error_type *config_error = config_error_alloc();

    {
        config_error_type *error_copy = config_error_alloc_copy(config_error);

        test_assert_true(config_error_equal(config_error, error_copy));
        test_assert_ptr_not_equal(config_error, error_copy);

        config_error_free(error_copy);
    }

    config_error_free(config_error);
    exit(0);
}
