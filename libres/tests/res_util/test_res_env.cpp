#include "catch2/catch.hpp"
#include <filesystem>
#include <fstream>

#include <ert/res_util/res_env.hpp>

namespace fs = std::filesystem;

TEST_CASE("res_env_alloc_PATH_executable", "[res_util]") {
    WITH_TMPDIR;

    SECTION("Absolute path that exists") {
        fs::path testpath = "/bin/sh";
        REQUIRE(res_env_alloc_PATH_executable(testpath) == testpath);
    }

    SECTION("Absolute path that does not exist") {
        fs::path testpath = "/does/not/exist/(hopefully)";
        REQUIRE(res_env_alloc_PATH_executable(testpath) == nullptr);
    }

    GIVEN("Script in relative path") {
        auto abspath = fs::current_path() / "script";
        {
            std::ofstream s{abspath};
            s << "#!/bin/sh\n";
            s << "echo Hei\n";
        }

        // chmod 0777 script
        fs::permissions(abspath, fs::perms::all);

        THEN("Absolute path is returned") {
            REQUIRE(res_env_alloc_PATH_executable(abspath) == abspath);
        }

        THEN("Relative path with ./") {
            REQUIRE(res_env_alloc_PATH_executable("./script") == abspath);
        }

        THEN("Relative path that does not start with ./ does not work") {
            // Due to backwards compatibility with previous (incorrect) ERT
            // behaviour
            auto relpath =
                fs::path("..") / fs::current_path().filename() / "script";
            // TODO: Check if filename is empty
            REQUIRE(fs::current_path().filename() != "");
            REQUIRE(res_env_alloc_PATH_executable("figure it out") == nullptr);
        }
    }

    SECTION("Path that exists") {
        fs::path testpath = "sh";
        auto expect = res_env_alloc_PATH_executable(testpath);
        REQUIRE(expect.filename() == "sh");
        REQUIRE(expect.is_absolute());
        REQUIRE(expect.status().permission() & fs::perm::owner_exec);
    }
}
