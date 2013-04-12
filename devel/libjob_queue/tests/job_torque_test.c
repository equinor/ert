/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
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
#include <stdio.h>
#include <stdbool.h>

#include <ert/util/test_util.h>
#include <ert/job_queue/torque_driver.h>

void test_option(torque_driver_type * driver, const char * option, const char * value) {
  test_assert_true(torque_driver_set_option(driver, option, value));
  test_assert_string_equal(torque_driver_get_option(driver, option), value);
}

void setoption_setalloptions_optionsset() {
  torque_driver_type * driver = torque_driver_alloc();

  test_option(driver, TORQUE_QSUB_CMD, "XYZaaa");
  test_option(driver, TORQUE_QSTAT_CMD, "xyZfff");
  test_option(driver, TORQUE_QDEL_CMD, "ZZyfff");

  printf("Options OK\n");
}

void getoption_nooptionsset_defaultoptionsreturned() {

}

int main(int argc, char ** argv) {
  getoption_nooptionsset_defaultoptionsreturned();
  setoption_setalloptions_optionsset();
  exit(0);
}
