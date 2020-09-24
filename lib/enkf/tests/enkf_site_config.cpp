/*
   Copyright (C) 2013  Equinor ASA, Norway.

   The file 'enkf_site_config.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/util/util.h>

#include <ert/enkf/site_config.hpp>


#define INCLUDE_KEY "INCLUDE"
#define DEFINE_KEY  "DEFINE"


void test_init(const char * config_file) {
  site_config_type * site_config = site_config_alloc_load_user_config(NULL);
  site_config_free( site_config );
}

int main(int argc , char ** argv) {
  const char * site_config_file = argv[1];

  util_install_signals();

  test_init( site_config_file );

  exit(0);
}

