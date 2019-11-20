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
    break;
  case FIELD:
    return "FIELD";
    break;
  case GEN_KW:
    return "GEN_KW";
    break;
  case SUMMARY:
    return "SUMMARY";
    break;
  case GEN_DATA:
    return "GEN_DATA";
    break;
  case EXT_PARAM:
    return "EXT_PARAM";
    break;
  default:
    util_abort("%s: internal error - unrecognized implementation type: %d - aborting \n",__func__ , impl_type);
    return NULL;
  }
}



#define if_strcmp(s) if (strcmp(impl_type_string , #s) == 0) impl_type = s
static ert_impl_type enkf_types_get_impl_type__(const char * impl_type_string) {
  ert_impl_type impl_type;
  if_strcmp(SUMMARY);
  else if_strcmp(FIELD);
  else if_strcmp(GEN_KW);
  else if_strcmp(GEN_DATA);
  else if_strcmp(EXT_PARAM);
  else impl_type = INVALID;
  return impl_type;
}
#undef if_strcmp
