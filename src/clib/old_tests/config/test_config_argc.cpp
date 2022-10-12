#include <signal.h>
#include <stdlib.h>

#include <ert/util/test_util.hpp>
#include <ert/util/util.hpp>

#include <ert/config/config_parser.hpp>

void install_SIGNALS(void) {
    signal(
        SIGSEGV,
        util_abort_signal); /* Segmentation violation, i.e. overwriting memory ... */
    signal(SIGINT, util_abort_signal); /* Control C */
    signal(
        SIGTERM,
        util_abort_signal); /* If killing the program with SIGTERM (the default kill signal) you will get a backtrace.
                                             Killing with SIGKILL (-9) will not give a backtrace.*/
}

int main(int argc, char **argv) {
    install_SIGNALS();
    {
        const char *argc_OK = argv[1];
        const char *argc_less = argv[2];
        const char *argc_more = argv[3];

        config_parser_type *config = config_alloc();
        config_schema_item_type *schema_item =
            config_add_schema_item(config, "ITEM", false);
        config_schema_item_set_argc_minmax(schema_item, 2, 2);

        {
            config_content_type *content =
                config_parse(config, argc_OK, "--", NULL, NULL, NULL,
                             CONFIG_UNRECOGNIZED_ERROR, true);
            test_assert_true(config_content_is_valid(content));
            config_content_free(content);
        }

        {
            config_content_type *content =
                config_parse(config, argc_less, "--", NULL, NULL, NULL,
                             CONFIG_UNRECOGNIZED_ERROR, true);
            test_assert_false(config_content_is_valid(content));

            {
                const config_error_type *config_error =
                    config_content_get_errors(content);
                const char *error_msg =
                    "Error when parsing config_file:\"argc_less\" Keyword:ITEM "
                    "must have at least 2 arguments.";

                test_assert_int_equal(config_error_count(config_error), 1);
                test_assert_string_equal(config_error_iget(config_error, 0),
                                         error_msg);
            }
            config_content_free(content);
        }

        {
            config_content_type *content =
                config_parse(config, argc_more, "--", NULL, NULL, NULL,
                             CONFIG_UNRECOGNIZED_ERROR, true);
            test_assert_false(config_content_is_valid(content));
            {
                const config_error_type *config_error =
                    config_content_get_errors(content);
                const char *error_msg =
                    "Error when parsing config_file:\"argc_more\" Keyword:ITEM "
                    "must have maximum 2 arguments.";

                test_assert_int_equal(config_error_count(config_error), 1);
                test_assert_string_equal(config_error_iget(config_error, 0),
                                         error_msg);
            }
            config_content_free(content);
        }

        config_free(config);
        exit(0);
    }
}
