#ifndef  __ECL_STATIC_KW_H__
#define  __ECL_STATIC_KW_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>
#include <ecl_kw.h>
#include <enkf_state.h>
#include <enkf_macros.h>
typedef struct ecl_static_kw_struct ecl_static_kw_type;


int                  ecl_static_kw_get_report_step(const ecl_static_kw_type * );
void                 ecl_static_kw_inc_counter(ecl_static_kw_type * , bool , int);
int                  ecl_static_kw_get_counter(const ecl_static_kw_type * );
ecl_static_kw_type * ecl_static_kw_alloc();
void                 ecl_static_kw_free(ecl_static_kw_type *ecl_static_kw);
void                 ecl_static_kw_init(ecl_static_kw_type * , const ecl_kw_type * );
ecl_kw_type        * ecl_static_kw_ecl_kw_ptr(const ecl_static_kw_type * );
void               * ecl_static_kw_alloc__(const void *);

SAFE_CAST_HEADER(ecl_static_kw);
VOID_FREE_HEADER(ecl_static_kw);
VOID_FREE_DATA_HEADER(ecl_static_kw);
VOID_FWRITE_HEADER(ecl_static_kw);
VOID_FREAD_HEADER(ecl_static_kw);
VOID_COPYC_HEADER(ecl_static_kw);
VOID_REALLOC_DATA_HEADER(ecl_static_kw);
VOID_ECL_WRITE_HEADER(ecl_static_kw);

#ifdef __cplusplus
}
#endif
#endif
