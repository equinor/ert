#include <ert/enkf/ert_run_context.hpp>

#include "catch2/catch.hpp"
#include <fmt/format.h>

SCENARIO("create run context", "[ert_run_context]") {

    GIVEN("An setup for run args") {
        const int iter = 7;
        const int ensemble_size = 10;

        std::vector<bool> active(ensemble_size, true);
        std::vector<std::string> runpaths{
            "/tmp/path/0000/7", "/tmp/path/0001/7", "/tmp/path/0002/7",
            "/tmp/path/0003/7", "/tmp/path/0004/7", "/tmp/path/0005/7",
            "/tmp/path/0006/7", "/tmp/path/0007/7", "/tmp/path/0008/7",
            "/tmp/path/0009/7",
        };
        std::vector<std::string> jobs{
            "Job0", "Job1", "Job2", "Job3", "Job4",
            "Job5", "Job6", "Job7", "Job8", "Job9",
        };
        enkf_fs_type *fs = nullptr;
        WHEN("some realizations are inactive") {
            active[6] = false;
            active[8] = false;
            THEN("created context has consistent getters") {
                ert_run_context_type *context =
                    ert_run_context_alloc_ENSEMBLE_EXPERIMENT(
                        fs, active, runpaths, jobs, iter);

                REQUIRE(ert_run_context_get_size(context) == 10);

                run_arg_type *run_arg0 = ert_run_context_iget_arg(context, 0);

                REQUIRE(run_arg_get_iter(run_arg0) == iter);
                REQUIRE("/tmp/path/0000/7" ==
                        std::string(run_arg_get_runpath(run_arg0)));

                for (int i = 0; i < active.size(); i++) {
                    if (!active[i]) {
                        REQUIRE(ert_run_context_iget_arg(context, i) ==
                                nullptr);
                    }
                }
                AND_WHEN("Some realizations are deactivated") {
                    ert_run_context_deactivate_realization(context, 0);
                    ert_run_context_deactivate_realization(context, 5);
                    ert_run_context_deactivate_realization(context, 9);
                    THEN("those realizations are inactive") {
                        REQUIRE(!ert_run_context_iactive(context, 0));
                        REQUIRE(!ert_run_context_iactive(context, 5));
                        REQUIRE(!ert_run_context_iactive(context, 9));
                    }
                }
                ert_run_context_free(context);
            }
        }
    }
}
