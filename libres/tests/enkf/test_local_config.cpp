#include <string>
#include <vector>

#include <ert/enkf/local_config.hpp>
#include "catch2/catch.hpp"

TEST_CASE("local_config", "[enkf]") {
    GIVEN("A default created LocalConfig with three observations and three "
          "parameters") {
        std::vector<std::string> obs_keys = {"OBS1", "OBS2", "OBS3"};
        std::vector<std::string> parameter_keys = {"PARAM1", "PARAM2",
                                                   "PARAM3"};
        LocalConfig local_config(parameter_keys, obs_keys);

        THEN("default_works") {
            const auto *global_ministep1 = local_config.global_ministep();
            const auto *global_ministep2 = local_config.ministep("ALL_ACTIVE");
            REQUIRE(*global_ministep1 == *global_ministep2);
            REQUIRE(global_ministep1->num_active_data() == 3);

            const auto *global_obsdata1 = local_config.global_obsdata();
            const auto *global_obsdata2 = local_config.obsdata("ALL_OBS");
            REQUIRE(*global_obsdata1 == *global_obsdata2);
            REQUIRE(global_obsdata1->size() == 3);

            auto &updatestep = local_config.updatestep();
            REQUIRE(updatestep.size() == 1);
            auto &ministep0 = updatestep[0];
            REQUIRE(ministep0 == *global_ministep1);
        }

        WHEN("adding_ministep") {
            auto &ministep = local_config.make_ministep("MINISTEP");
            auto &updatestep = local_config.updatestep();
            REQUIRE(updatestep.size() == 0);

            updatestep.add_ministep(ministep);
            REQUIRE(updatestep.size() == 1);

            auto &ministep0 = updatestep[0];
            REQUIRE(ministep == ministep0);

            updatestep.add_ministep(*local_config.global_ministep());
            REQUIRE(updatestep.size() == 2);
        }
    }
}
