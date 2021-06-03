/*
   Copyright (C) 2012  Equinor ASA, Norway.

   The file 'job_lsb_test.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/util/test_util.hpp>

#include <ert/job_queue/lsf_job_stat.hpp>
#include <ert/job_queue/lsf_driver.hpp>
#include <ert/job_queue/lsb.hpp>


void test_server(lsf_driver_type * driver , const char * server, lsf_submit_method_enum submit_method) {
  lsf_driver_set_option(driver , LSF_SERVER , server );
  test_assert_true( lsf_driver_get_submit_method( driver ) == submit_method );
}



/*
  This test should ideally be run twice in two different environments;
  with and without dlopen() access to the lsf libraries.
*/

int main( int argc , char ** argv) {
  lsf_submit_method_enum submit_NULL;
  lsb_type * lsb = lsb_alloc();
  if (lsb_ready(lsb))
    submit_NULL = LSF_SUBMIT_INTERNAL;
  else
    submit_NULL = LSF_SUBMIT_LOCAL_SHELL;


  test_server( driver , NULL        , submit_NULL );
  test_server( driver , "LoCaL"     , LSF_SUBMIT_LOCAL_SHELL );
  test_server( driver , "LOCAL"     , LSF_SUBMIT_LOCAL_SHELL );
  test_server( driver , "XLOCAL"    , LSF_SUBMIT_REMOTE_SHELL );
  test_server( driver , NULL        , submit_NULL );
  test_server( driver , "NULL"      , submit_NULL );
  test_server( driver , "be-grid01" , LSF_SUBMIT_REMOTE_SHELL );
  printf("Servers OK\n");

  lsb_free( lsb );
  exit(0);
}
