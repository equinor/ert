#ifndef __FIELD_H__
#define __FIELD_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <fortio.h>
#include <ecl_kw.h>
#include <enkf_macros.h>
#include <enkf_util.h>
#include <field_config.h>
#include <enkf_serialize.h>

typedef struct field_struct field_type;



int          field_get_global_index(const field_type * , int  , int  , int );
void         field_ijk_set(field_type * , int  , int  , int  , const void * );
void         field_indexed_set(field_type * field, ecl_type_enum , int , const int * , const void * );
void         field_indexed_add(field_type * field, ecl_type_enum , int , const int * , const void * );
double       field_iget_double(const field_type * , int );
double       field_ijk_get_double(const field_type * field, int  , int  , int );
float       field_iget_float(const field_type * , int );
float        field_ijk_get_float(const field_type * field, int  , int  , int );
void          field_ijk_get(const field_type * , int , int  , int , void *);
bool         field_ijk_valid(const field_type * , int , int , int );
void         field_ijk_get_if_valid(const field_type * , int  , int  , int , void * , bool *);
void 	     field_ecl_write(const field_type * , const char * , fortio_type *);
void 	     field_ecl_write1D_fortio(const field_type * , fortio_type *);
void 	     field_ecl_write3D_fortio(const field_type * , fortio_type *);
void         field_ROFF_export(const field_type * , const char * );
void 	     field_copy_ecl_kw_data(field_type * , const ecl_kw_type * );
field_type * field_alloc_shared(const field_config_type * , void * , int );
field_type * field_alloc(const field_config_type * );
void         field_free(field_type *);
void         field_get_dims(const field_type *, int *, int *, int *);
void         field_fload(field_type * , const char * , bool );
void         field_fload_auto(field_type * , const char * , bool );
void         field_export3D(const field_type * , void *, bool , ecl_type_enum , void *);
void         field_export(const field_type * , const char * , field_file_format_type );
field_type * field_copyc(const field_type *);
bool         field_cmp(const field_type *  , const field_type * );

double     * field_indexed_get_alloc(const field_type *, int, const int *);
void       * field_get_data(field_type * );
void         field_apply(field_type *  , field_func_type * );
void         field_inplace_output_transform(field_type * field);

void          field_iscale(field_type * , double );
void          field_isqrt(field_type *);
void          field_isqr(field_type *);
void          field_iaddsqr(field_type * , const field_type *);
void          field_iadd(field_type * , const field_type *);
void          field_imul_add(field_type * , double , const field_type *);
ecl_kw_type * field_alloc_ecl_kw_wrapper(const field_type * );


SAFE_CAST_HEADER(field);
VOID_ALLOC_HEADER(field);
VOID_FREE_HEADER(field);
VOID_FREE_DATA_HEADER(field);
VOID_REALLOC_DATA_HEADER(field);
VOID_COPYC_HEADER      (field);
VOID_SERIALIZE_HEADER  (field);
VOID_DESERIALIZE_HEADER (field);
VOID_INITIALIZE_HEADER(field);

VOID_ECL_WRITE_HEADER (field);
VOID_FWRITE_HEADER (field);
VOID_FREAD_HEADER  (field);
VOID_CLEAR_HEADER (field);
VOID_ECL_LOAD_HEADER(field);
VOID_USER_GET_HEADER(field);

#ifdef __cplusplus
}
#endif
#endif
