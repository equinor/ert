#include <stdlib.h>
#include <util.h>
#include <ecl_static_kw.h>
#include <ecl_kw.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <buffer.h>


struct ecl_static_kw_struct {
  int           __type_id;
  ecl_kw_type * ecl_kw;
};




ecl_static_kw_type * ecl_static_kw_alloc( ) {  
  ecl_static_kw_type * static_kw = util_malloc(sizeof *static_kw , __func__);
  static_kw->ecl_kw    	   = NULL;
  static_kw->__type_id 	   = STATIC;
  return static_kw;
}

/*
  ptr is pure dummy to satisfy the api.
*/
void * ecl_static_kw_alloc__(const void *ptr) {  
  return ecl_static_kw_alloc( );
}


void ecl_static_kw_ecl_write(const ecl_static_kw_type * ecl_static, const char * run_path /* Not used*/  , const char * path /* Not used */, fortio_type * fortio) {
  ecl_kw_fwrite(ecl_static->ecl_kw , fortio);
}


ecl_kw_type * ecl_static_kw_ecl_kw_ptr(const ecl_static_kw_type * ecl_static) { return ecl_static->ecl_kw; }


ecl_static_kw_type * ecl_static_kw_copyc(const ecl_static_kw_type *src) {
  ecl_static_kw_type * new = ecl_static_kw_alloc();
  if (src->ecl_kw != NULL)
    new->ecl_kw = ecl_kw_alloc_copy(src->ecl_kw);
  return new;
}


void ecl_static_kw_free_data(ecl_static_kw_type * kw) {
  if (kw->ecl_kw != NULL) ecl_kw_free(kw->ecl_kw);
  kw->ecl_kw = NULL;
}


void ecl_static_kw_free(ecl_static_kw_type * kw) {
  ecl_static_kw_free_data(kw);
  free(kw);
}


void ecl_static_kw_init(ecl_static_kw_type * ecl_static_kw, const ecl_kw_type * ecl_kw) {
  if (ecl_static_kw->ecl_kw != NULL)
    util_abort("%s: internal error: trying to assign ecl_kw to ecl_static_kw which is already set.\n",__func__);
  
  ecl_static_kw->ecl_kw = ecl_kw_alloc_copy(ecl_kw);
}


void ecl_static_kw_fread(ecl_static_kw_type * ecl_static_kw , FILE * stream) {
  enkf_util_fread_assert_target_type(stream , STATIC);
  if (ecl_static_kw->ecl_kw != NULL)
    util_abort("%s: internal error: trying to assign ecl_kw to ecl_static_kw which is already set.\n",__func__);
  ecl_static_kw->ecl_kw = ecl_kw_fread_alloc_compressed(stream);
}


bool ecl_static_kw_fwrite(const ecl_static_kw_type * ecl_static_kw , FILE * stream , bool internal_state) {
  enkf_util_fwrite_target_type(stream , STATIC);
  ecl_kw_fwrite_compressed(ecl_static_kw->ecl_kw , stream);
  return true;
}



void ecl_static_kw_load(ecl_static_kw_type * ecl_static_kw , buffer_type * buffer) {
  enkf_util_assert_buffer_type( buffer , STATIC );
  if (ecl_static_kw->ecl_kw != NULL)
    util_abort("%s: internal error: trying to assign ecl_kw to ecl_static_kw which is already set.\n",__func__);
  ecl_static_kw->ecl_kw = ecl_kw_buffer_alloc( buffer );
}


bool ecl_static_kw_store(const ecl_static_kw_type * ecl_static_kw , buffer_type * buffer, bool internal_state) {
  buffer_fwrite_int( buffer , STATIC );
  ecl_kw_buffer_store( ecl_static_kw->ecl_kw , buffer);
  return true;
}



/**

This is a pure dummy, memory is handled differently for this object
type:

  o The ecl_kw_type instance holding the data is boostrapped on fread().
  
  o The whole ecl_kw_type instances is free'd on free_data.

*/

void ecl_static_kw_realloc_data(ecl_static_kw_type * ecl_static_kw) {
  return ;
}




/*****************************************************************/
SAFE_CAST(ecl_static_kw , STATIC)
VOID_FREE(ecl_static_kw)
VOID_FREE_DATA(ecl_static_kw)
VOID_FWRITE (ecl_static_kw)
VOID_ECL_WRITE (ecl_static_kw)
VOID_FREAD  (ecl_static_kw)
VOID_COPYC(ecl_static_kw)
VOID_REALLOC_DATA(ecl_static_kw)
VOID_LOAD(ecl_static_kw)
VOID_STORE(ecl_static_kw)
