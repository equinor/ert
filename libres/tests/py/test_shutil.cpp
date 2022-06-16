#include <catch2/catch.hpp>

#include <ert/py/shutil.hpp>

TEST_CASE("which", "[shutil]") {
    SECTION("Absolute paths") {
        REQUIRE(ertpy::which("/bin/sh") == "/bin/sh");
    }
}
