#ifndef __MULTFLT_H__
#define __MULTFLT_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <multflt_config.h>
#include <enkf_util.h>
#include <enkf_macros.h>

typedef struct multflt_struct multflt_type;


void             multflt_output_transform(const multflt_type * );
void             multflt_get_output_data(const multflt_type * , double * );
const double   * multflt_get_output_ref(const multflt_type * );
const double   * multflt_get_data_ref(const multflt_type * );
void             multflt_get_data(const multflt_type * , double * );
void             multflt_set_data(multflt_type * , const double * );
multflt_type   * multflt_alloc(const multflt_config_type * );
void             multflt_free(multflt_type *);
void             multflt_ens_write(const multflt_type * , const char *);
void             multflt_ens_read(multflt_type * , const char *);
void             multflt_truncate(multflt_type * );
multflt_type   * multflt_alloc_mean(int , const multflt_type **);
const char     * multflt_get_name(const multflt_type * , int );
void             multflt_upgrade_103(const char * filename);


SAFE_CAST_HEADER(multflt);
SAFE_CONST_CAST_HEADER(multflt);
VOID_USER_GET_HEADER(multflt)
VOID_FREE_DATA_HEADER(multflt)
VOID_ECL_WRITE_HEADER  (multflt)
VOID_COPYC_HEADER      (multflt);
VOID_SERIALIZE_HEADER  (multflt);
VOID_DESERIALIZE_HEADER  (multflt);
VOID_INITIALIZE_HEADER(multflt);
VOID_FREE_HEADER(multflt);
VOID_ALLOC_HEADER(multflt);
VOID_REALLOC_DATA_HEADER(multflt);
VOID_STORE_HEADER(multflt);
VOID_LOAD_HEADER(multflt)
VOID_MATRIX_SERIALIZE_HEADER(multflt)
VOID_MATRIX_DESERIALIZE_HEADER(multflt)
       VOID_SET_INFLATION_HEADER(multflt);
       VOID_CLEAR_HEADER(multflt);
       VOID_IADD_HEADER(multflt);
       VOID_SCALE_HEADER(multflt);
       VOID_IMUL_HEADER(multflt);
       VOID_IADDSQR_HEADER(multflt);
       VOID_ISQRT_HEADER(multflt);

#ifdef __cplusplus
}
#endif
#endif
