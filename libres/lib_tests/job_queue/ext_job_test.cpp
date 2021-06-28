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
#include <sys/stat.h>

#include <ext/json/cJSON.h>

#include <ert/util/test_util.hpp>
#include <ert/util/test_work_area.hpp>

#include <ert/res_util/subst_list.hpp>
#include <ert/job_queue/ext_job.hpp>


void test_angular() {
  subst_list_type * subst_list = subst_list_alloc(NULL);
  {
    FILE * stream = util_fopen("ANGULAR", "w");
    fprintf(stream,"EXECUTABLE script\n");
    fprintf(stream, "EXECUTABLE script\n");
    fprintf(stream, "ENV VAR0 <VALUE0>\n");
    fprintf(stream, "EXEC_ENV NOT_SET\n");
    fprintf(stream, "EXEC_ENV VAR1 <NOT_SET>\n");
    fprintf(stream, "EXEC_ENV VAR2 VALUE\n");
    fclose(stream);
  }
  {
    ext_job_type * ext_job = ext_job_fscanf_alloc("ANGULAR", NULL, false, "ANGULAR", false);
    {
      FILE * stream = util_fopen("angular.json", "w");
      ext_job_json_fprintf(ext_job, 0, stream, subst_list);
      fclose(stream);
    }
    cJSON *json;
    {
      int buffer_size;
      char * buffer = util_fread_alloc_file_content("angular.json", &buffer_size);
      json = cJSON_Parse(buffer);
      {
        cJSON * env = cJSON_GetObjectItem(json, "environment");
        test_assert_true( env->type == cJSON_NULL );
      }

      {
        cJSON * exec_env = cJSON_GetObjectItem(json, "exec_env");
        test_assert_true( exec_env->type == cJSON_Object );
        test_assert_NULL( cJSON_GetObjectItem( exec_env, "VAR1"));
        test_assert_not_NULL( cJSON_GetObjectItem( exec_env, "VAR2"));
        {
          cJSON * value = cJSON_GetObjectItem(exec_env, "VAR2");
          test_assert_string_equal("VALUE", value->valuestring);
        }
        {
          cJSON * not_set = cJSON_GetObjectItem(exec_env, "NOT_SET");
          test_assert_true( not_set->type == cJSON_NULL);
        }
      }

      {
        cJSON * exec_env = cJSON_GetObjectItem(json, "exec_env");
        test_assert_true( exec_env->type == cJSON_Object );
        test_assert_NULL( cJSON_GetObjectItem( exec_env, "VAR1"));
        test_assert_not_NULL( cJSON_GetObjectItem( exec_env, "VAR2"));
        {
          cJSON * value = cJSON_GetObjectItem(exec_env, "VAR2");
          test_assert_string_equal("VALUE", value->valuestring);
        }
      }

      free(buffer);
    }
    cJSON_Delete(json);
    ext_job_free(ext_job);
  }
  subst_list_free(subst_list);
}



int main( int argc , char ** argv) {
  ecl::util::TestArea ta("joblist");
  {
    FILE * stream = util_fopen("script", "w");
    fprintf(stream, "Dummy script");
    fclose(stream);
    chmod("script", 0777);
  }
  test_angular();
}
