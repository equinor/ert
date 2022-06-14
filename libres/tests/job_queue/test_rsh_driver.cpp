#include <string>

#include "catch2/catch.hpp"

#include "../tmpdir.hpp"

std::pair<std::string, int> rsh_split_host(std::string hostname);

TEST_CASE("parse hostnames", "[rsh]") {
    SECTION("Only hostname") {
        auto [host, max_running] = rsh_split_host("foo");
        REQUIRE(host == "foo");
        REQUIRE(max_running == 1);
    }

    SECTION("Hostname with max_running") {
        auto [host, max_running] = rsh_split_host("bar:42");
        REQUIRE(host == "bar");
        REQUIRE(max_running == 42);
    }

    SECTION("Hostname with multiple colons") {
        auto [host, max_running] = rsh_split_host("foo:bar:quz:100");
        REQUIRE(host == "foo:bar:quz");
        REQUIRE(max_running == 100);
    }

    SECTION("max_running not an integer") {
        REQUIRE_THROWS(std::runtime_error, rsh_split_host("foo:bar"));
    }
}
