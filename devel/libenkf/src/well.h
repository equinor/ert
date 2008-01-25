#ifndef __WELL_H__
#define __WELL_H__
#include <ecl_sum.h>
#include <enkf_macros.h>
#include <enkf_util.h>

typedef struct well_struct well_type;

double    well_get(const well_type * , const char * );
void      well_load_summary_data(well_type * , int , const ecl_sum_type * );


MATH_OPS_HEADER(well);
VOID_ALLOC_HEADER(well);
VOID_FREE_HEADER(well);
VOID_FREE_DATA_HEADER(well);
VOID_REALLOC_DATA_HEADER(well);
VOID_COPYC_HEADER      (well);
VOID_SWAPIN_HEADER(well)
VOID_SWAPOUT_HEADER(well)
VOID_SERIALIZE_HEADER  (well)
VOID_DESERIALIZE_HEADER  (well)
VOID_FWRITE_HEADER (well)
VOID_FREAD_HEADER  (well)


VOID_FUNC_HEADER       (well_isqrt    );



#endif
