#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <enkf_types.h>


const char * enkf_types_get_impl_name(enkf_impl_type impl_type) {
  switch(impl_type) {
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
  case PGBOX:
    return "PGBOX";
    break;
  case GEN_KW:
    return "GEN_KW";
    break;
  default:
    fprintf(stderr,"%s: internal error - unrecognized implementation type: %d - aborting \n",__func__ , impl_type);
    abort();
  }
}








