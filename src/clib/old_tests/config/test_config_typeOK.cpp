#include <stdlib.h>

#include <ert/config/config_parser.hpp>
#include <ert/util/test_util.hpp>

int main(int argc, char **argv) {
    const char *config_file = argv[1];
    config_parser_type *config = config_alloc();
    {
        config_schema_item_type *item =
            config_add_schema_item(config, "TYPE_KEY", false);
        config_schema_item_set_argc_minmax(item, 4, 4);
        config_schema_item_iset_type(item, 0, CONFIG_INT);
        config_schema_item_iset_type(item, 1, CONFIG_FLOAT);
        config_schema_item_iset_type(item, 2, CONFIG_BOOL);

        item = config_add_schema_item(config, "SHORT_KEY", false);
        config_schema_item_set_argc_minmax(item, 1, 1);

        item = config_add_schema_item(config, "LONG_KEY", false);
        config_schema_item_set_argc_minmax(item, 3, CONFIG_DEFAULT_ARG_MAX);
    }
    {
        config_content_type *content =
            config_parse(config, config_file, "--", NULL, NULL, NULL,
                         CONFIG_UNRECOGNIZED_IGNORE, true);
        test_assert_true(config_content_is_valid(content));
        config_content_free(content);
    }

    exit(0);
}
