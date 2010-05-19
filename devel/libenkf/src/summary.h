#ifndef __SUMMARY_H__
#define __SUMMARY_H__
#include <ecl_sum.h>
#include <ecl_file.h>
#include <enkf_macros.h>
#include <enkf_util.h>
#include <summary_config.h>
#include <util.h>

double    summary_get(const summary_type * );
void      summary_upgrade_103( const char * filename );

UTIL_SAFE_CAST_HEADER(summary);
UTIL_SAFE_CAST_HEADER_CONST(summary);
VOID_ALLOC_HEADER(summary);
VOID_FREE_HEADER(summary);
VOID_COPY_HEADER(summary);
VOID_ECL_LOAD_HEADER(summary);
VOID_USER_GET_HEADER(summary);
VOID_STORE_HEADER(summary);
VOID_LOAD_HEADER(summary);
VOID_MATRIX_SERIALIZE_HEADER(summary)
VOID_MATRIX_DESERIALIZE_HEADER(summary)
VOID_SET_INFLATION_HEADER(summary);
VOID_CLEAR_HEADER(summary);
VOID_IADD_HEADER(summary);
VOID_SCALE_HEADER(summary);
VOID_IMUL_HEADER(summary);
VOID_IADDSQR_HEADER(summary);
VOID_ISQRT_HEADER(summary);
#endif
