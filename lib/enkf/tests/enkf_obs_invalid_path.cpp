/*
   Copyright (C) 2018  Equinor ASA, Norway.

   The file 'enkf_obs_invalid_file.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/
#include <stdlib.h>
#include <stdio.h>

#include <ert/util/test_work_area.h>
#include <ert/util/test_util.h>

#include <ert/enkf/enkf_obs.hpp>


void test_invalid_path() {
  test_work_area_type * work_area = test_work_area_alloc__("conf", true);
  util_make_path("obs_path");
  {
    FILE * stream = util_fopen("obs_path/conf.txt","w");
    fprintf(stream,
            "GENERAL_OBSERVATION WPR_DIFF_1 {"
            "DATA       = SNAKE_OIL_WPR_DIFF;"
            "INDEX_LIST = 400,800,1200,1800;"
            "RESTART    = 199;"
            "OBS_FILE   = obs_path/obs.txt;"
            "};");
    fclose(stream);
  }
  {
    FILE * stream = util_fopen("obs_path/obs.txt","w");
    fclose(stream);
  }

  conf_class_type * enkf_conf_class = enkf_obs_get_obs_conf_class();
  conf_instance_type * enkf_conf = conf_instance_alloc_from_file(enkf_conf_class,
                                                                 "enkf_conf",
                                                                 "obs_path/conf.txt");
  test_assert_true(conf_instance_get_path_error(enkf_conf));
  test_assert_false(conf_instance_validate(enkf_conf));

  conf_instance_free(enkf_conf);

  test_work_area_free(work_area);
}


void test_valid_path() {
  test_work_area_type * work_area = test_work_area_alloc("conf");
  util_make_path("obs_path");
  {
    FILE * stream = util_fopen("obs_path/conf.txt","w");
    fprintf(stream,
            "GENERAL_OBSERVATION WPR_DIFF_1 {\n"
            "DATA       = SNAKE_OIL_WPR_DIFF;\n"
            "INDEX_LIST = 400,800,1200,1800;\n"
            "RESTART    = 199;\n"
            "OBS_FILE   = obs.txt;\n"
            "};");
    fclose(stream);
  }
  {
    FILE * stream = util_fopen("obs_path/obs.txt","w");
    fclose(stream);
  }

  conf_class_type * enkf_conf_class = enkf_obs_get_obs_conf_class();
  conf_instance_type * enkf_conf = conf_instance_alloc_from_file(enkf_conf_class,
                                                                 "enkf_conf",
                                                                 "obs_path/conf.txt");

  test_assert_false(conf_instance_get_path_error(enkf_conf));
  test_assert_true(conf_instance_validate(enkf_conf));

  conf_instance_free(enkf_conf);
  test_work_area_free(work_area);
}



int main(int argc , char ** argv) {
  test_valid_path();
  test_invalid_path();
}
