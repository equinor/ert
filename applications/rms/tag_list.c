/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'tag_list.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/rms/rms_file.hpp>
#include <ert/rms/rms_tagkey.hpp>
#include <ert/rms/rms_stats.hpp>



int main (int argc , char **argv) {
  int i;

  argc--;
  argv++;

  for (i = 0; i < argc; i++) {
    rms_file_type *file = rms_file_alloc(argv[i] , false);
    rms_file_fread(file);
    rms_file_fprintf(file , stdout);
    rms_file_free(file);
  }

  return 0;
}

