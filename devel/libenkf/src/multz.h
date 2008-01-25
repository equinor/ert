
#ifndef __MULTZ_H__
#define __MULTZ_H__
#include <enkf_util.h>
#include <multz_config.h>


typedef struct multz_struct multz_type;

void             multz_output_transform(const multz_type * );
void             multz_get_output_data(const multz_type * , double * );
void             multz_get_data(const multz_type * , double * );
void             multz_set_data(multz_type * , const double * );
multz_type     * multz_alloc(const multz_config_type * );
void             multz_free(multz_type *);
char           * multz_alloc_ensname(const multz_type *);
char           * multz_alloc_eclname(const multz_type *);
void             multz_ecl_write(const multz_type * , const char *);
/*void             multz_direct_ecl_write(const multz_type * , const char *);*/
void             multz_ens_write(const multz_type * , const char *);
void             multz_ens_read(multz_type * , const char *);
void             multz_sample(multz_type *);
void             multz_truncate(multz_type *);
multz_type     * multz_alloc_mean(int , const multz_type **);
void             multz_TEST(void);
void             multz_fread(multz_type * , FILE * );
void             multz_fwrite(const multz_type * , FILE * );
char           * multz_alloc_description(const multz_type * , int );


MATH_OPS_HEADER(multz);
ENSEMBLE_MULX_VECTOR_HEADER(multz);
ENSEMBLE_MULX_VECTOR_VOID_HEADER(multz);
VOID_ALLOC_HEADER(multz);
VOID_FREE_HEADER(multz);
VOID_FREE_DATA_HEADER(multz);
VOID_REALLOC_DATA_HEADER(multz);
VOID_COPYC_HEADER      (multz);
VOID_SWAPIN_HEADER(multz)
VOID_SWAPOUT_HEADER(multz)
VOID_SERIALIZE_HEADER  (multz)
VOID_DESERIALIZE_HEADER  (multz)
VOID_ECL_WRITE_HEADER (multz)
VOID_FWRITE_HEADER (multz)
VOID_FREAD_HEADER  (multz)

VOID_FUNC_HEADER       (multz_sample   );
VOID_FUNC_HEADER       (multz_isqrt    );

#endif
