#include <stdlib.h>

#include <ert/util/test_util.hpp>
#include <ert/util/util.hpp>

#include <ert/config/config_parser.hpp>

void test_define(config_parser_type *config, const char *config_file) {
    hash_type *pre_defined_kw_map = hash_alloc();
    hash_insert_string(pre_defined_kw_map, "<CONFIG_FILE>", "TEST_VALUE");
    config_content_type *content =
        config_parse(config, config_file, NULL, NULL, "DEFINE",
                     pre_defined_kw_map, CONFIG_UNRECOGNIZED_IGNORE, true);
    hash_free(pre_defined_kw_map);
    test_assert_true(config_content_is_valid(content));
    {
        const subst_list_type *define_list =
            config_content_get_define_list(content);
        test_assert_true(subst_list_has_key(define_list, "VAR1"));
        test_assert_true(subst_list_has_key(define_list, "VAR2"));
        test_assert_true(subst_list_has_key(define_list, "VARX"));
        test_assert_true(subst_list_has_key(define_list, "<CONFIG_FILE>"));
        test_assert_false(subst_list_has_key(define_list, "VARY"));

        test_assert_string_equal(subst_list_get_value(define_list, "VAR1"),
                                 "100");
        test_assert_string_equal(subst_list_get_value(define_list, "VAR2"),
                                 "10");
        test_assert_string_equal(subst_list_get_value(define_list, "VARX"),
                                 "1");
        test_assert_string_equal(
            subst_list_get_value(define_list, "<CONFIG_FILE>"), "TEST_VALUE");
    }

    config_content_free(content);
}

config_parser_type *config_create_schema() {
    config_parser_type *config = config_alloc();

    config_add_schema_item(config, "SET", true);
    config_add_schema_item(config, "NOTSET", false);

    return config;
}

int main(int argc, char **argv) {
    util_install_signals();
    {
        const char *config_file = argv[1];
        config_parser_type *config = config_create_schema();

        test_define(config, config_file);

        config_free(config);
        exit(0);
    }
}
