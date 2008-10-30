#ifndef __SCALAR_H__
#define __SCALAR_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <enkf_util.h>
#include <scalar_config.h>
#include <enkf_serialize.h>
#include <stdio.h>


typedef struct scalar_struct scalar_type;

void             scalar_truncate(scalar_type * );
void             scalar_transform(scalar_type * );
void             scalar_get_data(const scalar_type * , double * );
void             scalar_get_output_data(const scalar_type * , double * );
void             scalar_set_data(scalar_type * , const double * );
char           * scalar_alloc_ensfile(const scalar_type * , const char *);
scalar_type      * scalar_alloc(const scalar_config_type * );
void             scalar_free(scalar_type *);
char           * scalar_alloc_ensname(const scalar_type *);
char           * scalar_alloc_eclname(const scalar_type *);
void             scalar_ecl_write(const scalar_type * , const char *);
void             scalar_ens_write(const scalar_type * , const char *);
void             scalar_ens_read(scalar_type * , const char *);
void             scalar_sample(scalar_type *);
void             scalar_truncate(scalar_type *);
void             scalar_TEST(void);
int              scalar_serialize(const scalar_type * , serial_state_type * , size_t , serial_vector_type *);
void             scalar_deserialize(scalar_type * , serial_state_type * , const serial_vector_type *);
int              scalar_serialize_part(const scalar_type * , serial_state_type * , bool , int , int , size_t , serial_vector_type *);
void             scalar_deserialize_part(scalar_type * , serial_state_type * , bool , int , int, const serial_vector_type *);
void             scalar_stream_fwrite(const scalar_type * scalar , FILE * );
void             scalar_stream_fread(scalar_type * scalar , FILE * );
void             scalar_realloc_data(scalar_type * scalar);
void             scalar_clear(scalar_type * scalar); 
const double   * scalar_get_output_ref(const scalar_type * );
const double   * scalar_get_data_ref  (const scalar_type * );
void             scalar_memcpy(scalar_type * , const scalar_type * );
void             scalar_free_data(scalar_type *);

MATH_OPS_HEADER(scalar);
VOID_ALLOC_HEADER(scalar);
VOID_FREE_HEADER(scalar);
VOID_FREE_DATA_HEADER(scalar);
VOID_REALLOC_DATA_HEADER(scalar);
VOID_COPYC_HEADER      (scalar);
VOID_ECL_WRITE_HEADER (scalar)
VOID_INITIALIZE_HEADER(scalar);

#ifdef __cplusplus
}
#endif
#endif
