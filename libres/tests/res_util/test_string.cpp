#include <array>
#include <set>
#include <list>

#include "catch2/catch.hpp"
#include <ert/res_util/string.hpp>

using namespace std::string_literals;
using namespace std::string_view_literals;

TEST_CASE("String splitting", "[res_util]") {
    GIVEN("Empty string") {
        std::string str;

        THEN("Callable is never called") {
            int num_called{};
            ert::split(str, ':', [&](auto) { ++num_called; });

            REQUIRE(num_called == 0);
        }
    }

    GIVEN("String without delimiter character") {
        std::string str = "hello world";

        THEN("Callable is called once") {
            int num_called{};
            ert::split(str, ':', [&](auto) { ++num_called; });

            REQUIRE(num_called == 1);
        }

        THEN("Parameter is equal to the whole string") {
            ert::split(str, ':', [&](auto x) { REQUIRE(x == str); });
        }
    }

    GIVEN("String with delimiter characters") {
        std::string str = "foo:bar baz:qux";

        THEN("Callable is called for each split part") {
            int num_called{};
            ert::split(str, ':', [&](auto x) {
                REQUIRE(num_called < 3);
                switch (num_called) {
                case 0:
                    REQUIRE(x == "foo");
                    break;
                case 1:
                    REQUIRE(x == "bar baz");
                    break;
                case 2:
                    REQUIRE(x == "qux");
                    break;
                }
                ++num_called;
            });

            REQUIRE(num_called == 3);
        }
    }

    GIVEN("String with consecutive delimiter characters") {
        std::string str = ":::::";

        THEN("Called is called five+1 times, with empty strings") {
            int num_called{};
            ert::split(str, ':', [&](auto x) {
                REQUIRE(x == "");
                ++num_called;
            });
            REQUIRE(num_called == 6);
        }
    }
}

TEST_CASE("Back element", "[res_util]") {
    GIVEN("Empty string") {
        std::string str;

        THEN("Return empty string") {
            auto res = ert::back_element(str, '.');
            REQUIRE(str == res);
        }
    }

    GIVEN("String with no delimiters") {
        std::string str = "hello world";

        THEN("Return same string") {
            auto res = ert::back_element(str, '.');
            REQUIRE(str == res);
        }
    }

    GIVEN("String with multiple delimiters") {
        std::string str = "hello.world.test";

        THEN("Return last part") {
            auto res = ert::back_element(str, '.');
            REQUIRE(res == "test");
        }
    }

    GIVEN("String only delimiters") {
        std::string str = "....";

        THEN("Return empty string") {
            auto res = ert::back_element(str, '.');
            REQUIRE(res == "");
        }
    }

    GIVEN("String where delimiter is the last character") {
        std::string str = "test.";

        THEN("Return empty string") {
            auto res = ert::back_element(str, '.');
            REQUIRE(res == "");
        }
    }
}

TEST_CASE("Join string", "[res_util]") {
    GIVEN("An empty std::array") {
        std::array<std::string, 0> strs;

        THEN("Joining produces empty string") {
            REQUIRE(ert::join(strs, "foo") == "");
        }
    }

    GIVEN("An std::array with a single element") {
        std::array<std::string, 1> strs{"foo"};

        THEN("Joining produces string with no seperator") {
            REQUIRE(ert::join(strs, "bar") == "foo");
        }
    }

    GIVEN("An std::array of strings") {
        std::array strs{"foo", "bar", "quiz"};

        THEN("Joining produces correct result") {
            REQUIRE(ert::join(strs, "") == "foobarquiz");
            REQUIRE(ert::join(strs, ",") == "foo,bar,quiz");
            REQUIRE(ert::join(strs, " :: ") == "foo :: bar :: quiz");
        }
    }

    GIVEN("An std::vector of std::string_view") {
        std::vector strs{"foo"sv, "bar"sv, "quiz"sv};

        THEN("Joining produces correct result") {
            REQUIRE(ert::join(strs, "") == "foobarquiz");
            REQUIRE(ert::join(strs, ",") == "foo,bar,quiz");
            REQUIRE(ert::join(strs, " :: ") == "foo :: bar :: quiz");
        }
    }

    GIVEN("An std::list of const char*") {
        std::list strs{"foo", "bar", "quiz"};

        THEN("Joining produces correct result") {
            REQUIRE(ert::join(strs, "") == "foobarquiz");
            REQUIRE(ert::join(strs, ",") == "foo,bar,quiz");
            REQUIRE(ert::join(strs, " :: ") == "foo :: bar :: quiz");
        }
    }

    GIVEN("An std::set of std::string") {
        std::set strs{"foo"s, "bar"s, "quiz"s};

        // std::set is a sorted container, so the iteration happens in
        // lexicographical order, which is different from the other joins.
        THEN("Joining produces correct result") {
            REQUIRE(ert::join(strs, "") == "barfooquiz");
            REQUIRE(ert::join(strs, ",") == "bar,foo,quiz");
            REQUIRE(ert::join(strs, " :: ") == "bar :: foo :: quiz");
        }
    }
}
