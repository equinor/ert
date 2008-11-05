#ifndef __FIELD_ACTIVE_H__
#define __FIELD_ACTIVE_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <enkf_macros.h>

typedef struct field_active_struct field_active_type;

SAFE_CAST_HEADER(field_active);

int  	   field_active_get_active_size( const field_active_type * );
int  	 * field_active_alloc_list_copy(const field_active_type * );



#ifdef __cplusplus
}
#endif
#endif
