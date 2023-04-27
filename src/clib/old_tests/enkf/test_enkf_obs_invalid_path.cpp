#include <stdio.h>
#include <stdlib.h>

#include <ert/util/test_util.h>
#include <ert/util/test_work_area.hpp>

#include <ert/enkf/enkf_obs.hpp>

void test_invalid_path() {
    ecl::util::TestArea ta("conf");
    util_make_path("obs_path");
    {
        FILE *stream = util_fopen("obs_path/conf.txt", "w");
        fprintf(stream, "GENERAL_OBSERVATION WPR_DIFF_1 {"
                        "DATA       = SNAKE_OIL_WPR_DIFF;"
                        "INDEX_LIST = 400,800,1200,1800;"
                        "RESTART    = 199;"
                        "OBS_FILE   = obs_path/obs.txt;"
                        "};");
        fclose(stream);
    }
    {
        FILE *stream = util_fopen("obs_path/obs.txt", "w");
        fclose(stream);
    }

    auto enkf_conf_class = enkf_obs_get_obs_conf_class();
    auto enkf_conf = conf_instance_alloc_from_file(enkf_conf_class, "enkf_conf",
                                                   "obs_path/conf.txt");
    test_assert_true(conf_instance_get_path_error(enkf_conf));
    test_assert_false(conf_instance_validate(enkf_conf));
}

void test_valid_path() {
    ecl::util::TestArea ta("valid");
    util_make_path("obs_path");
    {
        FILE *stream = util_fopen("obs_path/conf.txt", "w");
        fprintf(stream, "GENERAL_OBSERVATION WPR_DIFF_1 {\n"
                        "DATA       = SNAKE_OIL_WPR_DIFF;\n"
                        "INDEX_LIST = 400,800,1200,1800;\n"
                        "RESTART    = 199;\n"
                        "OBS_FILE   = obs.txt;\n"
                        "};");
        fclose(stream);
    }
    {
        FILE *stream = util_fopen("obs_path/obs.txt", "w");
        fclose(stream);
    }

    auto enkf_conf_class = enkf_obs_get_obs_conf_class();
    auto enkf_conf = conf_instance_alloc_from_file(enkf_conf_class, "enkf_conf",
                                                   "obs_path/conf.txt");

    test_assert_false(conf_instance_get_path_error(enkf_conf));
    test_assert_true(conf_instance_validate(enkf_conf));
}

int main(int argc, char **argv) {
    test_valid_path();
    test_invalid_path();
}
