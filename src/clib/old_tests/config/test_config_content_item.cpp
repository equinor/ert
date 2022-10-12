#include <stdlib.h>

#include <ert/util/hash.hpp>
#include <ert/util/test_util.hpp>

#include <ert/config/config_parser.hpp>

int main(int argc, char **argv) {
    const char *config_file = argv[1];
    config_parser_type *config = config_alloc();

    config_add_schema_item(config, "SET", true);
    config_add_schema_item(config, "NOTSET", false);

    {
        config_content_type *content =
            config_parse(config, config_file, "--", "INCLUDE", NULL, NULL,
                         CONFIG_UNRECOGNIZED_IGNORE, true);
        test_assert_true(config_content_is_valid(content));

        test_assert_true(config_content_has_item(content, "SET"));
        test_assert_false(config_content_has_item(content, "NOTSET"));
        test_assert_false(config_content_has_item(content, "UNKNOWN"));

        test_assert_true(config_has_schema_item(config, "SET"));
        test_assert_true(config_has_schema_item(config, "NOTSET"));
        test_assert_false(config_has_schema_item(config, "UNKNOWN"));

        config_content_free(content);
    }

    exit(0);
}
