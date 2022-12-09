#include <stdlib.h>

#include <ert/util/test_util.hpp>

#include <ert/config/config_parser.hpp>

void error(const char *msg) {
    fprintf(stderr, "%s", msg);
    exit(1);
}

int main(int argc, char **argv) {
    const char *config_file = argv[1];
    config_parser_type *config = config_alloc();
    {
        config_schema_item_type *item =
            config_add_schema_item(config, "TYPES_KEY", false);
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

        if (config_content_is_valid(content)) {
            error("Parse error\n");
        } else {
            if (content->parse_errors.size() > 0) {
                int i;
                for (auto error : content->parse_errors) {
                    printf("Error: %s \n", error.c_str());
                }
            }
            test_assert_int_equal(5, content->parse_errors.size());
        }
        config_content_free(content);
    }
    printf("OK \n");
    exit(0);
}
