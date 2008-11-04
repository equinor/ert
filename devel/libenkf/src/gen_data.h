#ifndef __GEN_PARAM_H__
#define __GEN_PARAM_H__
#include <gen_param_config.h>
#include <ecl_sum.h>
#include <enkf_macros.h>

typedef struct gen_param_struct gen_param_type;

void          	         gen_param_ecl_load(gen_param_type * , const char *  , const char * , const ecl_sum_type * , int );
gen_param_type 	       * gen_param_alloc(const gen_param_config_type *);
void          	         gen_param_free(gen_param_type * );
double                   gen_param_iget_double(const gen_param_type * , int );
gen_param_config_type   * gen_param_get_config(const gen_param_type * );
int                      gen_param_get_size(const gen_param_type * );



VOID_ALLOC_HEADER(gen_param);
VOID_FREE_HEADER(gen_param);
VOID_FREE_DATA_HEADER(gen_param);
VOID_REALLOC_DATA_HEADER(gen_param);
VOID_COPYC_HEADER      (gen_param);
VOID_FWRITE_HEADER (gen_param)
VOID_FREAD_HEADER  (gen_param)
VOID_ECL_WRITE_HEADER(gen_param);
VOID_SERIALIZE_HEADER(gen_param);
VOID_DESERIALIZE_HEADER(gen_param);
VOID_INITIALIZE_HEADER(gen_param);

#endif
