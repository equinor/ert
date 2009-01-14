#ifndef __EQUIL_H__
#define __EQUIL_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <enkf_macros.h>
#include <equil_config.h>
#include <enkf_util.h>

typedef struct equil_struct equil_type;

equil_type     * equil_copyc(const equil_type * );
equil_type     * equil_alloc(const equil_config_type * );
void             equil_free(equil_type *);
char           * equil_alloc_ensname(const equil_type *);
char           * equil_alloc_eclname(const equil_type *);


SAFE_CAST_HEADER(equil);
VOID_FREE_DATA_HEADER(equil)
VOID_SERIALIZE_HEADER(equil);
VOID_DESERIALIZE_HEADER(equil);
VOID_FREAD_HEADER      (equil)
VOID_FWRITE_HEADER     (equil)
VOID_INITIALIZE_HEADER       (equil);
VOID_FREE_HEADER       (equil);
VOID_COPYC_HEADER      (equil);
MATH_OPS_VOID_HEADER(equil);
VOID_ALLOC_HEADER(equil);
VOID_ECL_WRITE_HEADER(equil);


#ifdef __cplusplus
}
#endif
#endif
