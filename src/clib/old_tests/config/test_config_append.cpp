#include <stdlib.h>

#include <ert/util/test_util.hpp>

#include <ert/config/config_parser.hpp>

int main(int argc, char **argv) {
    const char *config_file = argv[1];
    config_parser_type *config = config_alloc();
    config_schema_item_type *item =
        config_add_schema_item(config, "APPEND", false);
    config_schema_item_set_argc_minmax(item, 1, 1);

    {
        config_content_type *content =
            config_parse(config, config_file, "--", NULL, NULL, NULL,
                         CONFIG_UNRECOGNIZED_IGNORE, true);
        test_assert_true(config_content_is_valid(content));
        test_assert_int_equal(config_content_get_occurences(content, "APPEND"),
                              3);

        {
            const char *value = config_content_get_value(content, "APPEND");
            test_assert_string_equal(value, "VALUE3");
        }

        config_content_free(content);
    }

    {
        config_content_type *content =
            config_parse(config, "DoesNotExist", "--", NULL, NULL, NULL,
                         CONFIG_UNRECOGNIZED_IGNORE, true);
        test_assert_false(config_content_is_valid(content));
        config_content_free(content);
    }
    exit(0);
}
