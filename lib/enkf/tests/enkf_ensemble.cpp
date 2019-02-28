/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'enkf_ensemble.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <stdio.h>
#include <unistd.h>

#include <ert/util/test_util.h>
#include <ert/util/util.h>
#include <ert/res_util/arg_pack.hpp>

#include <ert/config/config_parser.hpp>

#include <ert/ecl/ecl_sum.h>

#include <ert/enkf/ensemble_config.hpp>
#include <ert/enkf/time_map.hpp>







int main(int argc , char ** argv) {
  ensemble_config_type * ensemble = ensemble_config_alloc(NULL, NULL, NULL);
  ensemble_config_free( ensemble );
  exit(0);
}

