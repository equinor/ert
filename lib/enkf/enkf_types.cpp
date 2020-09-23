/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'enkf_types.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <string.h>

#include <ert/util/util.h>

#include <ert/enkf/enkf_types.hpp>


/*****************************************************************/


const char * enkf_types_get_impl_name(ert_impl_type impl_type) {
  switch(impl_type) {
  case(INVALID):
    return "INVALID";
  case FIELD:
    return "FIELD";
  case GEN_KW:
    return "GEN_KW";
  case SUMMARY:
    return "SUMMARY";
  case GEN_DATA:
    return "GEN_DATA";
  case EXT_PARAM:
    return "EXT_PARAM";
  default:
    util_abort("%s: internal error - unrecognized implementation type: %d - aborting \n",__func__ , impl_type);
    return NULL;
  }
}
