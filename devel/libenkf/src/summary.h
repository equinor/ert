#ifndef __SUMMARY_H__
#define __SUMMARY_H__
#include <ecl_sum.h>
#include <ecl_file.h>
#include <enkf_macros.h>
#include <enkf_util.h>
#include <summary_config.h>

double    summary_get(const summary_type * );
void      summary_upgrade_103( const char * filename );

SAFE_CAST_HEADER(summary);
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
VOID_STORE_HEADER(summary);
VOID_LOAD_HEADER(summary);
VOID_MATRIX_SERIALIZE_HEADER(summary)
VOID_MATRIX_DESERIALIZE_HEADER(summary)
#endif
