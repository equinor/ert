
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>

#include "catch2/catch.hpp"

#include "../tmpdir.hpp"

namespace fs = std::filesystem;
namespace detail {
std::vector<std::string> parse_hostnames(const char *);
}

TEST_CASE("parse hostnames", "[lsf]") {
    WITH_TMPDIR;
    fs::path file_path = fs::current_path() / "exclud_hosts";

    GIVEN("Empty stream") {
        std::ofstream stream{file_path};
        stream.close();

        auto hosts = detail::parse_hostnames(file_path.c_str());
        REQUIRE(hosts == std::vector<std::string>{});
    }

    GIVEN("Non-empty stream") {
        std::ofstream stream{file_path};
        stream << "hname1:4*hname2:13*st-rst666-01-42.st.example.org:1*hname4:"
                  "hname5\n";
        stream.close();

        auto hosts = detail::parse_hostnames(file_path.c_str());

        REQUIRE(hosts ==
                std::vector<std::string>{"hname1", "hname2",
                                         "st-rst666-01-42.st.example.org",
                                         "hname4", "hname5"});
    }
}
