#ifndef __GEN_KW_H__
#define __GEN_KW_H__

#include <gen_kw_config.h>
#include <enkf_util.h>
#include <enkf_macros.h>

typedef struct gen_kw_struct gen_kw_type;


void             gen_kw_output_transform(const gen_kw_type * );
void             gen_kw_get_output_data(const gen_kw_type * , double * );
const double   * gen_kw_get_output_ref(const gen_kw_type * );
const double   * gen_kw_get_data_ref(const gen_kw_type * );
void             gen_kw_get_data(const gen_kw_type * , double * );
void             gen_kw_set_data(gen_kw_type * , const double * );
void             gen_kw_sample(gen_kw_type *);
gen_kw_type   * gen_kw_alloc(const gen_kw_config_type * );
void             gen_kw_free(gen_kw_type *);
void             gen_kw_ecl_write(const gen_kw_type * , const char *);
/*void             gen_kw_direct_ecl_write(const gen_kw_type * , const char *);*/
void             gen_kw_ens_write(const gen_kw_type * , const char *);
void             gen_kw_ens_read(gen_kw_type * , const char *);
void             gen_kw_truncate(gen_kw_type * );
gen_kw_type   *  gen_kw_alloc_mean(int , const gen_kw_type **);
const char     * gen_kw_get_name(const gen_kw_type * , int );
void             gen_kw_filter_file(const gen_kw_type * , const char * );

VOID_ECL_WRITE_HEADER  (gen_kw)
VOID_FWRITE_HEADER  (gen_kw)
VOID_FREAD_HEADER   (gen_kw)
VOID_COPYC_HEADER      (gen_kw);
VOID_SWAPOUT_HEADER(gen_kw);
VOID_SWAPIN_HEADER(gen_kw);
VOID_SERIALIZE_HEADER  (gen_kw);
VOID_DESERIALIZE_HEADER  (gen_kw);


VOID_FUNC_HEADER       (gen_kw_sample   );
VOID_FUNC_HEADER       (gen_kw_free     );


MATH_OPS_HEADER(gen_kw);
VOID_ALLOC_HEADER(gen_kw);

#endif
