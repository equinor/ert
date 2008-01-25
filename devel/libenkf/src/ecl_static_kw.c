#include <stdlib.h>
#include <enkf_state.h>
#include <util.h>
#include <ecl_static_kw.h>
#include <ecl_static_kw_config.h>
#include <ecl_kw.h>


#define  DEBUG
#define  TARGET_TYPE STATIC
#include "enkf_debug.h"


struct ecl_static_kw_struct {
  DEBUG_DECLARE
  const ecl_static_kw_config_type * config;
  ecl_kw_type               * ecl_kw;
};



const ecl_kw_type * ecl_static_kw_ecl_kw_ptr(const ecl_static_kw_type * ecl_static) { return ecl_static->ecl_kw; }

ecl_static_kw_type * ecl_static_kw_alloc(const ecl_static_kw_config_type * config) {
  ecl_static_kw_type * static_kw = malloc(sizeof *static_kw);
  static_kw->ecl_kw = NULL;
  static_kw->config = config;
  DEBUG_ASSIGN(static_kw)
  return static_kw;
}


ecl_static_kw_type * ecl_static_kw_copyc(const ecl_static_kw_type *src) {
  ecl_static_kw_type * new = ecl_static_kw_alloc(src->config);
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
  ecl_static_kw->ecl_kw = ecl_kw_alloc_copy(ecl_kw);
}


void ecl_static_kw_fread(ecl_static_kw_type * ecl_static_kw , FILE * stream) {
  DEBUG_ASSERT(ecl_static_kw);
  enkf_util_fread_assert_target_type(stream , STATIC , __func__);
  ecl_kw_fread_realloc_compressed(ecl_static_kw->ecl_kw , stream);
}


void ecl_static_kw_fwrite(const ecl_static_kw_type * ecl_static_kw , FILE * stream) {
  DEBUG_ASSERT(ecl_static_kw);
  enkf_util_fwrite_target_type(stream , STATIC);
  ecl_kw_fwrite_compressed(ecl_static_kw->ecl_kw , stream);
}




void ecl_static_kw_swapout(ecl_static_kw_type * ecl_static_kw , FILE * stream) {
  DEBUG_ASSERT(ecl_static_kw);
  {
    ecl_static_kw_fwrite(ecl_static_kw , stream);
    ecl_static_kw_free_data(ecl_static_kw);
  }
}



void ecl_static_kw_swapin(ecl_static_kw_type * ecl_static_kw , FILE * stream) {
  DEBUG_ASSERT(ecl_static_kw);
  {
    ecl_static_kw_fread(ecl_static_kw , stream);
  }
}


VOID_SWAPIN(ecl_static_kw);
VOID_SWAPOUT(ecl_static_kw);
/*****************************************************************/
VOID_ALLOC(ecl_static_kw)
VOID_FREE(ecl_static_kw)
VOID_FREE_DATA(ecl_static_kw)
VOID_FWRITE (ecl_static_kw)
VOID_FREAD  (ecl_static_kw)
VOID_COPYC(ecl_static_kw)

