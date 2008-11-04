#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <enkf_types.h>
#include <util.h>


/*****************************************************************/

const char * enkf_types_get_impl_name(enkf_impl_type impl_type) {
  switch(impl_type) {
  case(INVALID):
    return "INVALID";
    break;
  case STATIC:
    return "STATIC";
    break;
  case MULTZ:
    return "MULTZ";
    break;
  case MULTFLT:
    return "MULTFLT";
    break;
  case EQUIL:
    return "EQUIL";
    break;
  case WELL:
    return "WELL";
    break;
  case FIELD:
    return "FIELD";
    break;
  case GEN_KW:
    return "GEN_KW";
    break;
  case RELPERM:
    return "RELPERM";
    break;
  case SUMMARY:
    return "SUMMARY";
    break;
  case HAVANA_FAULT:
    return "HAVANA_FAULT";
    break;
  case GEN_DATA:
    return "GEN_DATA";
    break;
  default:
    util_abort("%s: internal error - unrecognized implementation type: %d - aborting \n",__func__ , impl_type);
    return NULL;
  }
}


#define if_strcmp(s) if (strcmp(impl_type_string , #s) == 0) impl_type = s
static enkf_impl_type enkf_types_get_impl_type__(const char * impl_type_string) {
  enkf_impl_type impl_type;
  if_strcmp(STATIC);
  else if_strcmp(MULTZ);
  else if_strcmp(SUMMARY);
  else if_strcmp(HAVANA_FAULT);
  else if_strcmp(MULTFLT);
  else if_strcmp(EQUIL);
  else if_strcmp(FIELD);
  else if_strcmp(WELL);
  else if_strcmp(GEN_KW);
  else if_strcmp(RELPERM);
  else if_strcmp(GEN_DATA);
  else impl_type = INVALID;
  return impl_type;
}
#undef if_strcmp


enkf_impl_type enkf_types_get_impl_type(const char * __impl_type_string) {
  char * impl_type_string = util_alloc_string_copy(__impl_type_string);
  util_strupr(impl_type_string);  
  enkf_impl_type impl_type = enkf_types_get_impl_type__(impl_type_string);
  if (impl_type == INVALID) 
    util_abort("%s: enkf_type: %s not recognized - aborting \n",__func__ , __impl_type_string);
  
  free(impl_type_string);
  return impl_type;
}


/*
  This will return INVALIID if given an invalid
  input string - not fail.
*/
  
enkf_impl_type enkf_types_check_impl_type(const char * impl_type_string) {
  return enkf_types_get_impl_type__(impl_type_string);
}







