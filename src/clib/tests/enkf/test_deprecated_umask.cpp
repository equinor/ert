#include "catch2/catch.hpp"
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "../tmpdir.hpp"
#include <ert/config/config_parser.hpp>
#include <ert/enkf/site_config.hpp>
extern "C" void set_site_config(const char *site_config);
SCENARIO("Using UMASK in config") {
    GIVEN("A config file with umask set to 007") {
        WITH_TMPDIR;
        std::ofstream configfile("config.ert");
        configfile << "UMASK 007" << std::endl;
        configfile << "QUEUE_SYSTEM LOCAL" << std::endl;
        configfile << "NUM_REALIZATIONS 1" << std::endl;
        configfile.close();
        config_parser_type *config = config_alloc();
        config_content_type *content;

        site_config_add_config_items(config, false);

        WHEN("Parsing the config file") {
            content = config_parse(config, "config.ert", "--", NULL, NULL, NULL,
                                   CONFIG_UNRECOGNIZED_WARN, true);
            const stringlist_type *warnings =
                config_content_get_warnings(content);

            THEN("A warning is raised") {
                REQUIRE(stringlist_contains(
                    warnings,
                    "UMASK is deprecated and will be removed in the future."));
            }
        }
    }
    GIVEN("A config file with umask set to 0") {
        WITH_TMPDIR;
        std::ofstream configfile("config.ert");
        configfile << "UMASK 0" << std::endl;
        configfile << "QUEUE_SYSTEM LOCAL" << std::endl;
        configfile << "NUM_REALIZATIONS 1" << std::endl;
        configfile.close();
        config_parser_type *config = config_alloc();
        config_content_type *content;

        site_config_add_config_items(config, false);

        WHEN("Parsing the config file") {
            content = config_parse(config, "config.ert", "--", NULL, NULL, NULL,
                                   CONFIG_UNRECOGNIZED_WARN, true);
            const stringlist_type *warnings =
                config_content_get_warnings(content);

            THEN("A warning is raised") {
                REQUIRE(stringlist_contains(
                    warnings,
                    "UMASK is deprecated and will be removed in the future."));
            }
        }
    }
}
