/*
   Copyright (C) 2012  Equinor ASA, Norway.

   The file 'job_lsf_test.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <ert/util/util.hpp>

#include <ert/job_queue/lsf_job_stat.hpp>
#include <ert/job_queue/lsf_driver.hpp>


void test_option(lsf_driver_type * driver , const char * option , const char * value) {
  test_assert_true( lsf_driver_set_option( driver , option , value));
  test_assert_string_equal((const char *) lsf_driver_get_option( driver , option) , value);
}



void test_status(int lsf_status , job_status_type job_status) {
  test_assert_true( lsf_driver_convert_status( lsf_status ) == job_status);
}

void test_options(void) {
  lsf_driver_type * driver = (lsf_driver_type *) lsf_driver_alloc();
  test_assert_false( lsf_driver_has_project_code( driver ));

  // test setting values
  test_option( driver , LSF_BSUB_CMD    , "Xbsub");
  test_option( driver , LSF_BJOBS_CMD   , "Xbsub");
  test_option( driver , LSF_BKILL_CMD   , "Xbsub");
  test_option( driver , LSF_RSH_CMD     , "RSH");
  test_option( driver , LSF_LOGIN_SHELL , "shell");
  test_option( driver , LSF_BSUB_CMD    , "bsub");
  test_option( driver , LSF_PROJECT_CODE, "my-ppu");
  test_option( driver , LSF_BJOBS_TIMEOUT, "1234");

  test_assert_true( lsf_driver_has_project_code( driver ));

  // test unsetting/resetting options to default values
  test_option( driver , LSF_BSUB_CMD    , NULL);
  test_option( driver , LSF_BJOBS_CMD   , NULL);
  test_option( driver , LSF_BKILL_CMD   , NULL);
  test_option( driver , LSF_RSH_CMD     , NULL);
  test_option( driver , LSF_LOGIN_SHELL , NULL);
  test_option( driver , LSF_PROJECT_CODE, NULL);

  // Setting NULL to numerical options should leave the value unchanged
  lsf_driver_set_option( driver, LSF_BJOBS_TIMEOUT, NULL);
  test_assert_string_equal( (const char *) lsf_driver_get_option( driver, LSF_BJOBS_TIMEOUT ), "1234");

  lsf_driver_free( driver );
}

void test_status_tr() {
  test_status( JOB_STAT_PEND   , JOB_QUEUE_PENDING );
  test_status( JOB_STAT_PSUSP  , JOB_QUEUE_RUNNING );
  test_status( JOB_STAT_USUSP  , JOB_QUEUE_RUNNING );
  test_status( JOB_STAT_SSUSP  , JOB_QUEUE_RUNNING );
  test_status( JOB_STAT_RUN    , JOB_QUEUE_RUNNING );
  test_status( JOB_STAT_NULL   , JOB_QUEUE_NOT_ACTIVE );
  test_status( JOB_STAT_DONE   , JOB_QUEUE_DONE );
  test_status( JOB_STAT_EXIT   , JOB_QUEUE_EXIT );
  test_status( JOB_STAT_UNKWN  , JOB_QUEUE_EXIT );
  test_status( 192             , JOB_QUEUE_DONE );
}


void test_cmd(void) {
  const char * project_code = "XXX_PROJECT";
  lsf_driver_type * driver = (lsf_driver_type *) lsf_driver_alloc();
  {
    stringlist_type * cmd = lsf_driver_alloc_cmd( driver , "out" , "job", "/bin/echo" , 1 , 0 , NULL );
    test_assert_false( lsf_driver_has_project_code( driver ));
    test_assert_false( stringlist_contains( cmd , "-P" ));
    stringlist_free( cmd );
  }

  lsf_driver_set_option( driver , LSF_PROJECT_CODE , project_code);
  {
    stringlist_type * cmd = lsf_driver_alloc_cmd( driver , "out" , "job", "/bin/echo" , 1 , 0 , NULL );
    int P_index = stringlist_find_first( cmd , "-P" );
    test_assert_true( lsf_driver_has_project_code( driver ));
    test_assert_true( P_index >= 0 );
    test_assert_string_equal( stringlist_iget( cmd , P_index + 1) , project_code );
    stringlist_free( cmd );
  }

  lsf_driver_free(driver);
}


void test_submit_method() {
  lsf_driver_type * driver = (lsf_driver_type *) lsf_driver_alloc();
#ifdef HAVE_LSF_LIBRARY
  test_assert_int_not_equal( lsf_driver_get_submit_method(driver), LSF_SUBMIT_INVALID);
#else
  test_assert_int_equal(lsf_driver_get_submit_method(driver), LSF_SUBMIT_LOCAL_SHELL);
#endif
  lsf_driver_free(driver);
}

int main( int argc , char ** argv) {
  util_install_signals();

  test_options();
  test_status_tr( );
  test_cmd();
  test_submit_method();

  exit(0);
}
