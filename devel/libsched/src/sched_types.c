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


#define WCONHIST_STRING  "WCONHIST"
#define DATES_STRING     "DATES"
#define COMPDAT_STRING   "COMPDAT"
#define TSTEP_STRING     "TSTEP"
#define TIME_STRING      "TIME"
#define WELSPECS_STRING  "WELSPECS"
#define GRUPTREE_STRING  "GRUPTREE"
#define INCLUDE_STRING   "INCLUDE"
#define WCONINJ_STRING   "WCONINJ"
#define WCONINJE_STRING  "WCONINJE"
#define WCONINJH_STRING  "WCONINJH"
#define WCONPROD_STRING  "WCONPROD"

#define UNTYPED_STRING   "UNTYPED"



/**
   This function does a direct translation of a string name to
   implementation type - i.e. an enum instance. Observe that
   (currently) no case-normalization is performed.
*/

sched_kw_type_enum sched_kw_type_from_string(const char * kw_name)
{
  sched_kw_type_enum kw_type = UNTYPED;

  if     ( strcmp(kw_name, GRUPTREE_STRING ) == 0) kw_type = GRUPTREE ;
  else if( strcmp(kw_name, TSTEP_STRING    ) == 0) kw_type = TSTEP    ;
  else if( strcmp(kw_name, INCLUDE_STRING  ) == 0) kw_type = INCLUDE  ;
  else if( strcmp(kw_name, TIME_STRING     ) == 0) kw_type = TIME     ;
  else if( strcmp(kw_name, DATES_STRING    ) == 0) kw_type = DATES    ;
  else if( strcmp(kw_name, WCONHIST_STRING ) == 0) kw_type = WCONHIST ;
  else if( strcmp(kw_name, WELSPECS_STRING ) == 0) kw_type = WELSPECS ;
  else if( strcmp(kw_name, WCONINJ_STRING  ) == 0) kw_type = WCONINJ  ;
  else if( strcmp(kw_name, WCONINJE_STRING ) == 0) kw_type = WCONINJE ;
  else if( strcmp(kw_name, WCONINJH_STRING ) == 0) kw_type = WCONINJH ;
  else if( strcmp(kw_name, WCONPROD_STRING ) == 0) kw_type = WCONPROD ;
  else if( strcmp(kw_name, COMPDAT_STRING  ) == 0) kw_type = COMPDAT  ;   
  
  return kw_type;
}


const char * sched_kw_type_name(sched_kw_type_enum kw_type) {
  if      ( kw_type == GRUPTREE ) return GRUPTREE_STRING ;
  else if ( kw_type == TSTEP    ) return TSTEP_STRING    ;
  else if ( kw_type == INCLUDE  ) return INCLUDE_STRING  ;
  else if ( kw_type == TIME     ) return TIME_STRING     ;
  else if ( kw_type == DATES    ) return DATES_STRING    ;
  else if ( kw_type == WCONHIST ) return WCONHIST_STRING ;
  else if ( kw_type == WELSPECS ) return WELSPECS_STRING ;
  else if ( kw_type == WCONINJ  ) return WCONINJ_STRING  ;
  else if ( kw_type == WCONINJE ) return WCONINJE_STRING ;
  else if ( kw_type == WCONINJH ) return WCONINJH_STRING ;
  else if ( kw_type == WCONPROD ) return WCONPROD_STRING ;
  else if ( kw_type == COMPDAT  ) return COMPDAT_STRING  ;   

  return UNTYPED_STRING; /* Unknown type */
}

