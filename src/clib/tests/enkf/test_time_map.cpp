#include "catch2/catch.hpp"
#include <cppitertools/enumerate.hpp>
#include <filesystem>

#include "../tmpdir.hpp"
#include <ert/ecl/ecl_sum.hpp>
#include <ert/enkf/time_map.hpp>

TEST_CASE("TimeMap", "[time_map]") {
    GIVEN("ECLIPSE summary") {
        auto summary_path =
            LOCAL_TEST_DATA_DIR / "snake_oil/refcase/SNAKE_OIL_FIELD";
        auto summary = ecl_sum_fread_alloc_case(summary_path.c_str(), ":");
        REQUIRE(summary != nullptr);

        WHEN("Adding it to empty TimeMap") {
            TimeMap time_map;
            REQUIRE(time_map.summary_update(summary).empty());

            THEN("Adding again works") {
                REQUIRE(time_map.summary_update(summary).empty());
            }

            THEN("The TimeMap is populated") {
                time_t start_time = ecl_sum_get_start_time(summary);
                time_t end_time = ecl_sum_get_end_time(summary);
                REQUIRE(time_map.front() == start_time);
                REQUIRE(time_map.back() == end_time);
            }

            THEN("The index map is automorphic") {
                auto indices = time_map.indices(summary);
                REQUIRE(indices.size() ==
                        ecl_sum_get_last_report_step(summary) + 1);
                for (int i = ecl_sum_get_first_report_step(summary);
                     i < indices.size(); ++i)
                    REQUIRE(i == indices[i]);
            }
        }

        WHEN("Adding to a TimeMap with existing data") {
            TimeMap time_map{0, 1, 2, 3}; // Seconds in UNIX epoch

            THEN("summary_update fails due to inconsistencies") {
                REQUIRE(!time_map.summary_update(summary).empty());
            }
        }

        WHEN("Empty TimeMap") {
            TimeMap time_map;
            THEN("The index map is also empty") {
                REQUIRE(time_map.indices(summary).empty());
            }
        }

        WHEN("Adding it to existing TimeMap") {
            TimeMap time_map{0, 1, 2, 3};
            THEN("It errors") {
                REQUIRE(!time_map.summary_update(summary).empty());
            }
        }

        ecl_sum_free(summary);
    }
}
