#include <util.h>
#include <hash.h>
#include <vector.h>
#include <local_ministep.h>
#include <local_reportstep.h>
#include <enkf_macros.h>

/**
   One enkf update is described/configured by the data structure in
   local_ministep.c. This file implements a local report_step, which
   is a collection of ministeps - in many cases a local_reportstep will
   only consist of one single local_ministep; but in principle it can
   contain several.
*/

#define LOCAL_REPORTSTEP_TYPE_ID 77159

struct local_reportstep_struct {
  int           __type_id;
  char        * name;
  vector_type * ministep;
};



SAFE_CAST(local_reportstep , LOCAL_REPORTSTEP_TYPE_ID)


local_reportstep_type * local_reportstep_alloc( const char * name ) {
  local_reportstep_type * reportstep = util_malloc( sizeof * reportstep , __func__);

  reportstep->name      = util_alloc_string_copy( name );
  reportstep->ministep  = vector_alloc_new();
  reportstep->__type_id = LOCAL_REPORTSTEP_TYPE_ID;
  return reportstep;
}


/**
   Observe that use_count values are not copied. 
*/
local_reportstep_type * local_reportstep_alloc_copy( const local_reportstep_type * src , const char * name ) {
  local_reportstep_type * new = local_reportstep_alloc( name );
  for (int i = 0; i < vector_get_size(src->ministep ); i++)
    local_reportstep_add_ministep( new , vector_iget( src->ministep , i) );
  return new;
}
   


void local_reportstep_free( local_reportstep_type * reportstep) {
  free( reportstep->name );
  vector_free( reportstep->ministep );
}


void local_reportstep_free__(void * arg) {
  local_reportstep_type * reportstep = local_reportstep_safe_cast( arg );
  local_reportstep_free( reportstep );
}


void local_reportstep_add_ministep( local_reportstep_type * reportstep , local_ministep_type * ministep) {
  vector_append_ref( reportstep->ministep , ministep );   /* Observe that the vector takes NO ownership */
}



local_ministep_type * local_reportstep_iget_ministep( const local_reportstep_type * reportstep , int index) {
  return vector_iget( reportstep->ministep , index );
}


int local_reportstep_get_num_ministep( const local_reportstep_type * reportstep) {
  return vector_get_size( reportstep->ministep );
}

