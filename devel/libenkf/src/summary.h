#ifndef __SUMMARY_H__
#define __SUMMARY_H__
#include <ecl_sum.h>
#include <ecl_block.h>
#include <enkf_macros.h>
#include <enkf_util.h>

typedef struct summary_struct summary_type;

double    summary_get(const summary_type * );
void      summary_load_data(summary_type * , int , const ecl_block_type * , const ecl_sum_type * );


MATH_OPS_VOID_HEADER(summary);
VOID_ALLOC_HEADER(summary);
VOID_FREE_HEADER(summary);
VOID_FREE_DATA_HEADER(summary);
VOID_REALLOC_DATA_HEADER(summary);
VOID_COPYC_HEADER      (summary);
VOID_SERIALIZE_HEADER  (summary)
VOID_DESERIALIZE_HEADER  (summary)
VOID_FWRITE_HEADER (summary)
VOID_FREAD_HEADER  (summary)
VOID_ECL_LOAD_HEADER(summary);
VOID_USER_GET_HEADER(summary);
VOID_FPRINTF_RESULTS_HEADER(summary);
#endif
