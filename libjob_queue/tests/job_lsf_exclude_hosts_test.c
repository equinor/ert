/*
 * Copyright (C) 2016  Statoil ASA, Norway.
 *
 * This file is part of ERT - Ensemble based Reservoir Tool.
 *
 * ERT is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * ERT is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.
 *
 * See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
 * for more details.
 */

#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>

#include <assert.h>
#include <ert/util/util.h>

#include <ert/job_queue/lsf_driver.h>
#include <ert/job_queue/lsf_job_stat.h>

void test_submit(lsf_driver_type * driver, const char * cmd) {
  {
    char * node1 = "enern";
    char * node2 = "toern";
    char * black1 = "!hname=enern";
    char * black2 = "!hname=toern";

    lsf_driver_add_exclude_hosts(driver, node1);
    lsf_driver_add_exclude_hosts(driver, node2);

    {
      stringlist_type * argv = lsf_driver_alloc_cmd(driver, "", "NAME", "bsub", 1, 0, NULL);
      if (!stringlist_contains(argv, black1))
        exit(1);

      if (!stringlist_contains(argv, black2))
        exit(1);
    }
  }
}

int main(int argc, char ** argv) {
  lsf_driver_type * driver = lsf_driver_alloc();
  test_submit(driver, argv[1]);
  lsf_driver_free(driver);
  exit(0);
}
