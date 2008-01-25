#include <stdlib.h>
#include <rms_type.h>

/*****************************************************************/
/* A microscopic (purely internal) type object only used 
   for storing the hash type_map */
/*****************************************************************/




void rms_type_free(void *rms_t) {
  free( (__rms_type *) rms_t);
}


static __rms_type * rms_type_set(__rms_type *rms_t , rms_type_enum rms_type , int sizeof_ctype) {
  rms_t->rms_type     = rms_type;
  rms_t->sizeof_ctype = sizeof_ctype;
  return rms_t;
}


__rms_type * rms_type_alloc(rms_type_enum rms_type, int sizeof_ctype) {
  __rms_type *rms_t   = malloc(sizeof *rms_t);
  rms_type_set(rms_t , rms_type , sizeof_ctype);
  return rms_t;
}


const void * rms_type_copyc(const void *__rms_t) {
  const __rms_type *rms_t = (const __rms_type *) __rms_t;
  __rms_type *new_t = rms_type_alloc(rms_t->rms_type , rms_t->sizeof_ctype);
  return new_t;
}
