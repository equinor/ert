#ifndef __RELPERM_H__
#define __RELPERM_H__
#include <enkf_util.h>
#include <relperm_config.h>

typedef struct relperm_struct relperm_type;


relperm_type * relperm_alloc(const relperm_config_type * );
void           relperm_initialize(relperm_type *, int iens);
void           relperm_get_data(const relperm_type *, double *);
void           relperm_set_data(relperm_type *, const double *);
void           relperm_ecl_write(const relperm_type *,const char *);
void           relperm_output_transform(const relperm_type *);
const double * relperm_get_output_ref(const relperm_type *);
void relperm_get_output_data(const relperm_type *, double *);

#endif
