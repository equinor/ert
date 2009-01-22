#ifndef __RELPERM_H__
#define __RELPERM_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <enkf_util.h>
#include <relperm_config.h>

typedef struct relperm_struct relperm_type;


relperm_type * relperm_alloc(const relperm_config_type * );
void           relperm_get_data(const relperm_type *, double *);
void           relperm_set_data(const relperm_type *, const double *);
void           relperm_ecl_write_f90test(const relperm_type *,const double *, const char *);
void           relperm_output_transform(const relperm_type *);
const double * relperm_get_output_ref(const relperm_type *);
void           relperm_get_output_data(const relperm_type *, double *);

SAFE_CAST_HEADER(relperm)
MATH_OPS_VOID_HEADER(relperm);
ENSEMBLE_MULX_VECTOR_HEADER(relperm);
ENSEMBLE_MULX_VECTOR_VOID_HEADER(relperm);
VOID_ALLOC_HEADER(relperm);
VOID_FREE_HEADER(relperm);
VOID_FREE_DATA_HEADER(relperm);
VOID_REALLOC_DATA_HEADER(relperm);
VOID_COPYC_HEADER      (relperm);
VOID_SERIALIZE_HEADER  (relperm)
VOID_DESERIALIZE_HEADER  (relperm)
VOID_ECL_WRITE_HEADER (relperm)
VOID_FWRITE_HEADER (relperm)
VOID_FREAD_HEADER  (relperm)
VOID_INITIALIZE_HEADER(relperm)

#ifdef __cplusplus
}
#endif
#endif
