#include <fstream>
#include <stdlib.h>
#include <string>

#include <catch2/catch.hpp>

#include "../tmpdir.hpp"
#include <ert/enkf/enkf_obs.hpp>

std::shared_ptr<conf_class> enkf_obs_get_obs_conf_class();

/*
 * Write conf-file with given keywords, then parse it.
 * Returns pointer to the parsed configuration
 */
std::shared_ptr<conf_instance>
write_conf(std::vector<std::string> keyword_lines) {
    std::string buf("GENERAL_OBSERVATION WPR_DIFF_1 {\n");
    buf += "   DATA       = SNAKE_OIL_WPR_DIFF;\n";
    buf += "   INDEX_LIST = 400,800,1200,1800;\n";
    buf += "   RESTART    = 199;\n";
    for (auto s : keyword_lines)
        buf += s + "\n";
    buf += "};\n";

    std::ofstream stream("obs_path/conf.txt", std::ostream::out);
    stream.write(buf.data(), buf.size());
    stream.close(); // close before reading in next step!

    auto enkf_conf_class = enkf_obs_get_obs_conf_class();
    return conf_instance::from_file(enkf_conf_class, "enkf_conf",
                                    "obs_path/conf.txt");
}

/*
 * Syntactic sugar for checking if a string contains another string
 * The memmber-function ::contains() is introduced in C++23
 */
bool contains(std::string errors, std::string kw) {
    return errors.find(kw) != std::string::npos;
}

/**
 * Syntactic sugar for making sure a file exists ("touching" it)
 *
 * Note that std::ofstream is a proper RAII object which means
 * neither flush nor close is required.
 */
void touch_file(std::string name) {
    std::ofstream stream(name.data(), std::ostream::out);
}

TEST_CASE("Test parsing keywords in configuration", "[unittest]") {
    GIVEN("A workearea") {
        WITH_TMPDIR;
        std::filesystem::create_directory("obs_path");

        GIVEN("A conf file with no relevant keywords") {
            auto enkf_conf = write_conf({});
            THEN("There are no errors") {
                std::string errors = enkf_conf->get_path_error();
                REQUIRE(!contains(errors, "OBS_FILE"));
                REQUIRE(!contains(errors, "INDEX_FILE"));
            }
        }

        GIVEN("A conf file with OBS_FILE keyword in it") {
            auto enkf_conf = write_conf({"OBS_FILE   = obs.txt;"});
            WHEN("There is no OBS_FILE present") {
                THEN("Error refers to missing file") {
                    std::string errors = enkf_conf->get_path_error();
                    REQUIRE(contains(
                        errors, "OBS_FILE=>" +
                                    std::filesystem::current_path().native() +
                                    "/obs_path/obs.txt"));
                    REQUIRE(!contains(errors, "INDEX_FILE"));
                }
            }
            WHEN("The obs file is present") {
                touch_file("obs_path/obs.txt");
                THEN("There are no errors") {
                    std::string errors = enkf_conf->get_path_error();
                    REQUIRE(!contains(errors, "OBS_FILE"));
                    REQUIRE(!contains(errors, "INDEX_FILE"));
                }
            }
        }

        GIVEN("A conf file with OBS_FILE in subdir obs_path") {
            auto enkf_conf = write_conf({"OBS_FILE   = obs_path/obs.txt;"});

            WHEN("The OBS_FILE is in the wrong location") {
                touch_file("obs_path/obs.txt");
                THEN("Error refers to missing file") {
                    std::string errors = enkf_conf->get_path_error();
                    REQUIRE(contains(
                        errors, "OBS_FILE=>" +
                                    std::filesystem::current_path().native() +
                                    "/obs_path/obs_path/obs.txt"));
                    REQUIRE(!contains(errors, "INDEX_FILE"));
                }
            }
            WHEN("The OBS_FILE is in the subdirectory") {
                util_make_path("obs_path/obs_path");
                touch_file("obs_path/obs_path/obs.txt");
                THEN("There are no errors") {
                    std::string errors = enkf_conf->get_path_error();
                    REQUIRE(!contains(errors, "OBS_FILE"));
                    REQUIRE(!contains(errors, "INDEX_FILE"));
                }
            }
        }

        GIVEN("A conf file with OBS_FILE and INDEX_FILE keywords") {
            auto enkf_conf = write_conf({"OBS_FILE    = obs.txt;"
                                         "INDEX_FILE  = index.txt;"});

            WHEN("Only OBS_FILE exists") {
                touch_file("obs_path/obs.txt");
                THEN("Error refer to missing index file") {
                    std::string errors = enkf_conf->get_path_error();
                    REQUIRE(!contains(errors, "OBS_FILE"));
                    REQUIRE(contains(
                        errors, "INDEX_FILE=>" +
                                    std::filesystem::current_path().native() +
                                    "/obs_path/index.txt"));
                }
            }
            WHEN("obs, and index files exist") {
                touch_file("obs_path/obs.txt");
                touch_file("obs_path/index.txt");
                THEN("There are no errors") {
                    std::string errors = enkf_conf->get_path_error();
                    REQUIRE(!contains(errors, "OBS_FILE"));
                    REQUIRE(!contains(errors, "INDEX_FILE"));
                }
            }
        }
    }
}
