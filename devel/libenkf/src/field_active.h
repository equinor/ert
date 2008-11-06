#ifndef __FIELD_ACTIVE_H__
#define __FIELD_ACTIVE_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <enkf_macros.h>
#include <active_list.h>

typedef struct field_active_struct field_active_type;

SAFE_CAST_HEADER(field_active);

void field_active_update_active_list(const field_active_type *  , active_list_type * );


#ifdef __cplusplus
}
#endif
#endif
