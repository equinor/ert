#ifndef __GEN_KW_H__
#define __GEN_KW_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <fortio.h>
#include <gen_kw_config.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <enkf_serialize.h>

typedef struct gen_kw_struct gen_kw_type;


void             gen_kw_output_transform(const gen_kw_type * );
void             gen_kw_get_output_data(const gen_kw_type * , double * );
const double   * gen_kw_get_output_ref(const gen_kw_type * );
const double   * gen_kw_get_data_ref(const gen_kw_type * );
  //void             gen_kw_get_data(const gen_kw_type * , double * );
void             gen_kw_set_data(gen_kw_type * , const double * );

gen_kw_type    * gen_kw_alloc(const gen_kw_config_type * );

void             gen_kw_free(gen_kw_type *);
void             gen_kw_ens_write(const gen_kw_type * , const char *);
void             gen_kw_ens_read(gen_kw_type * , const char *);
void             gen_kw_truncate(gen_kw_type * );
gen_kw_type   *  gen_kw_alloc_mean(int , const gen_kw_type **);
const char     * gen_kw_get_name(const gen_kw_type * , int );
void             gen_kw_filter_file(const gen_kw_type * , const char * );
void             gen_kw_export(const gen_kw_type * , int * , char ***, double **);
void             gen_kw_ensemble_fprintf_results(const gen_kw_type ** , int , const char * );
void             gen_kw_upgrade_103( const char * filename );
char           * gen_kw_alloc_user_key(const gen_kw_config_type *  , const char * , int );

SAFE_CAST_HEADER(gen_kw);
ALLOC_STATS_HEADER(gen_kw)
VOID_ECL_WRITE_HEADER  (gen_kw)
VOID_FWRITE_HEADER  (gen_kw)
VOID_FREAD_HEADER   (gen_kw)
VOID_COPYC_HEADER      (gen_kw);
VOID_SERIALIZE_HEADER  (gen_kw);
VOID_DESERIALIZE_HEADER  (gen_kw);
VOID_FREE_DATA_HEADER(gen_kw)
VOID_INITIALIZE_HEADER(gen_kw);
VOID_FREE_HEADER       (gen_kw);
MATH_OPS_VOID_HEADER(gen_kw);
VOID_ALLOC_HEADER(gen_kw);
VOID_ECL_WRITE_HEADER(gen_kw);
VOID_REALLOC_DATA_HEADER(gen_kw);
VOID_FPRINTF_RESULTS_HEADER(gen_kw);
VOID_USER_GET_HEADER(gen_kw);
VOID_STORE_HEADER(gen_kw);
VOID_LOAD_HEADER(gen_kw);
VOID_MATRIX_SERIALIZE_HEADER(gen_kw)
VOID_MATRIX_DESERIALIZE_HEADER(gen_kw)
#ifdef __cplusplus
}
#endif
#endif
