#ifndef  __ECL_STATIC_KW_H__
#define  __ECL_STATIC_KW_H__
#include <ecl_kw.h>
#include <ecl_static_kw_config.h>
#include <enkf_state.h>
typedef struct ecl_static_kw_struct ecl_static_kw_type;


ecl_static_kw_type * ecl_static_kw_alloc(const ecl_static_kw_config_type *);
void                 ecl_static_kw_free(ecl_static_kw_type *ecl_static_kw);
void                 ecl_static_kw_init(ecl_static_kw_type * , const ecl_kw_type * );
const ecl_kw_type  * ecl_static_kw_ecl_kw_ptr(const ecl_static_kw_type * );

VOID_SWAPIN_HEADER(ecl_static_kw);
VOID_SWAPOUT_HEADER(ecl_static_kw);
VOID_ALLOC_HEADER(ecl_static_kw);
VOID_FREE_HEADER(ecl_static_kw);
VOID_FREE_DATA_HEADER(ecl_static_kw);
VOID_FWRITE_HEADER(ecl_static_kw);
VOID_FREAD_HEADER(ecl_static_kw);
VOID_COPYC_HEADER(ecl_static_kw);

#endif
