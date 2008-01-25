#ifndef  __ENKF_ECL_KW_H__
#define  __ENKF_ECL_KW_H__
#include <ecl_kw.h>
#include <enkf_util.h>
#include <enkf_state.h>
#include <enkf_ecl_kw_config.h>

typedef struct     enkf_ecl_kw_struct enkf_ecl_kw_type;

ecl_kw_type      * enkf_ecl_kw_alloc_ecl_kw(const enkf_ecl_kw_type * , bool  , bool );
void               enkf_ecl_kw_get_data(enkf_ecl_kw_type * , const ecl_kw_type *);
enkf_ecl_kw_type * enkf_ecl_kw_alloc(const enkf_ecl_kw_config_type * );
void               enkf_ecl_kw_free(enkf_ecl_kw_type *);
enkf_ecl_kw_type * enkf_ecl_kw_copyc(const enkf_ecl_kw_type * );

VOID_ALLOC_ENSFILE_HEADER(enkf_ecl_kw);
MATH_OPS_HEADER(enkf_ecl_kw);
VOID_ALLOC_HEADER(enkf_ecl_kw);
VOID_FREE_HEADER(enkf_ecl_kw);
VOID_FREE_DATA_HEADER(enkf_ecl_kw);
VOID_REALLOC_DATA_HEADER(enkf_ecl_kw);
VOID_ENS_WRITE_HEADER(enkf_ecl_kw);
VOID_ENS_READ_HEADER(enkf_ecl_kw);
VOID_COPYC_HEADER(enkf_ecl_kw);
VOID_SWAPIN_HEADER(enkf_ecl_kw)
VOID_SWAPOUT_HEADER(enkf_ecl_kw)

#endif
