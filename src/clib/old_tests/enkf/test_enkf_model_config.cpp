#include <stdlib.h>

#include <ert/util/test_util.h>

#include <ert/enkf/model_config.hpp>

void test_create() {
    model_config_type *model_config = model_config_alloc_empty();
    model_config_free(model_config);
}

void test_runpath() {
    model_config_type *model_config = model_config_alloc_empty();
    model_config_add_runpath(model_config, "KEY", "RunPath%d");
    model_config_add_runpath(model_config, "KEY2", "2-RunPath%d");
    test_assert_true(model_config_select_runpath(model_config, "KEY"));
    test_assert_false(model_config_select_runpath(model_config, "KEYX"));
    test_assert_string_equal("RunPath%d",
                             model_config_get_runpath_as_char(model_config));

    model_config_set_runpath(model_config, "PATH%d");
    test_assert_string_equal("PATH%d",
                             model_config_get_runpath_as_char(model_config));
    test_assert_true(model_config_select_runpath(model_config, "KEY2"));
    test_assert_string_equal("2-RunPath%d",
                             model_config_get_runpath_as_char(model_config));
    test_assert_true(model_config_select_runpath(model_config, "KEY"));
    test_assert_string_equal("PATH%d",
                             model_config_get_runpath_as_char(model_config));

    test_assert_false(model_config_runpath_requires_iter(model_config));
    model_config_set_runpath(model_config, "iens%d/iter%d");
    test_assert_true(model_config_runpath_requires_iter(model_config));

    model_config_free(model_config);
}

void test_export_file() {
    model_config_type *model_config = model_config_alloc_empty();

    model_config_free(model_config);
}

int main(int argc, char **argv) {
    test_create();
    test_runpath();
    test_export_file();
    exit(0);
}
