/*
   Copyright (C) 2018  Equinor ASA, Norway.

   The file 'ext_job_test.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <stdbool.h>
#include <sys/stat.h>

#include <ert/util/test_util.hpp>
#include <ert/util/test_work_area.hpp>
#include <ert/job_queue/ext_job.hpp>


void test_angular() {
  {
    FILE * stream = util_fopen("ANGULAR", "w");
    fprintf(stream,"EXECUTABLE script\n");
    fprintf(stream,"EXEC_ENV VAR <NOT_SET>\n");
    fclose(stream);
  }
  {
    ext_job_type * ext_job = ext_job_fscanf_alloc("ANGULAR", NULL, false, "ANGULAR", false);
    test_assert_false( ext_job_exec_env_is_set( ext_job, "VAR"));
    ext_job_free(ext_job);
  }
}

void test_NULL() {
  {
    FILE * stream = util_fopen("NULL", "w");
    fprintf(stream,"EXECUTABLE script\n");
    fprintf(stream,"EXEC_ENV EMPTY_VAR\n");
    fprintf(stream,"EXEC_ENV VAR null\n");
    fclose(stream);
  }
  {
    ext_job_type * ext_job = ext_job_fscanf_alloc("NULL", NULL, false, "NULL", false);
    test_assert_NULL( ext_job_exec_env_get( ext_job, "EMPTY_VAR"));
    test_assert_NULL( ext_job_exec_env_get( ext_job, "VAR"));
    ext_job_free(ext_job);
  }
}



int main( int argc , char ** argv) {
  test_work_area_type * test_area = test_work_area_alloc("ext_job");

  {
    FILE * stream = util_fopen("script", "w");
    fprintf(stream, "Dummy script");
    fclose(stream);
    chmod("script", 0777);
  }
  test_angular();
  test_NULL();

  test_work_area_free(test_area);
}
