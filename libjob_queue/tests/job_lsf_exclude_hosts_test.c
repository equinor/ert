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

void test_bjobs_parse_hosts() {
  const char* full_hostnames = "hname1:4*hname2:13*hname3:1*hname4:hname5\n";
  const char* hostnames = "hname1:hname2:hname3:hname4:hname5";
  stringlist_type * expected = stringlist_alloc_from_split(hostnames,":");
  if (stringlist_get_size(expected) != 5) {
    printf("Even expected has wrong size.\n");
    exit(1);
  }

  const char * fname = util_alloc_tmp_file("/tmp", "ert_job_exec_host", true);

  FILE * fptr;
  fptr = fopen(fname, "w");
  fprintf(fptr, full_hostnames); // : is std bjobs delimiter
  fclose(fptr);

  stringlist_type * hosts = lsf_job_alloc_parse_hostnames(fname);

  if (!stringlist_equal(expected, hosts)) {
    printf("hosts differ: expected [%s] got [%s]\n",
            stringlist_alloc_joined_string(expected, ":"),
            stringlist_alloc_joined_string(hosts,    ":"));
    exit(1);
  }

  util_unlink_existing(fname);
  free(fname);
  stringlist_free( hosts );
}

int main(int argc, char ** argv) {
  lsf_driver_type * driver = lsf_driver_alloc();
  test_submit(driver, argv[1]);
  lsf_driver_free(driver);
  test_bjobs_parse_hosts();
  exit(0);
}
