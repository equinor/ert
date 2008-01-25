#ifndef __PGBOX_H__
#define __PGBOX_H__
#include <enkf_util.h>
#include <enkf_macros.h>
#include <enkf_types.h>
#include <ecl_box.h>
#include <pgbox_config.h>
#include <field.h>

typedef struct pgbox_struct pgbox_type;


pgbox_type * pgbox_alloc(const pgbox_config_type *);
void         pgbox_apply(pgbox_type * );
void         pgbox_free(pgbox_type * );
void         pgbox_set_target_field(pgbox_type *  , field_type * );
void         pgbox_fwrite(const pgbox_type * , FILE * );
void         pgbox_fread(pgbox_type * , FILE * );

/*****************************************************************/

MATH_OPS_HEADER(pgbox);
ENSEMBLE_MULX_VECTOR_HEADER(pgbox);
VOID_ALLOC_HEADER(pgbox);
VOID_FREE_HEADER(pgbox);
VOID_FREE_DATA_HEADER(pgbox);
VOID_REALLOC_DATA_HEADER(pgbox);
VOID_COPYC_HEADER      (pgbox);
VOID_SWAPIN_HEADER(pgbox)
VOID_SWAPOUT_HEADER(pgbox)
VOID_SERIALIZE_HEADER  (pgbox);
VOID_DESERIALIZE_HEADER (pgbox);

VOID_ECL_WRITE_HEADER (pgbox)
VOID_FWRITE_HEADER (pgbox)
VOID_FREAD_HEADER  (pgbox)


VOID_FUNC_HEADER       (pgbox_sample   );
VOID_FUNC_HEADER       (pgbox_isqrt    );


#endif
