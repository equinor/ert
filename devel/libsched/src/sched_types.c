#include <string.h>
#include <util.h>
#include <sched_types.h>


#define ECL_DEFAULT_KW "*"
#define TYPE_WATER_STRING "WATER"
#define TYPE_GAS_STRING   "GAS"
#define TYPE_OIL_STRING   "OIL"

const char * sched_phase_type_string(sched_phase_type type) {
  switch (type) {
  case(WATER):
    return TYPE_WATER_STRING;
  case(GAS):
    return TYPE_GAS_STRING;
  case(OIL):
    return TYPE_OIL_STRING;
  default:
    return ECL_DEFAULT_KW;
  }
}

sched_phase_type sched_phase_type_from_string(const char * type_string) {
  if (strcmp(type_string , TYPE_WATER_STRING) == 0)
    return WATER;
  else if (strcmp(type_string , TYPE_GAS_STRING) == 0)
    return GAS;
  else if (strcmp(type_string , TYPE_OIL_STRING) == 0)
    return OIL;
  else {
    util_abort("%s: Could not recognize:%s as injector phase. Valid values are: [%s, %s, %s] \n",__func__ , type_string , TYPE_WATER_STRING , TYPE_GAS_STRING , TYPE_OIL_STRING);
    return 0;
  }
}


/*****************************************************************/
