#ifndef __LOCAL_UPDATESTEP_H__
#define __LOCAL_UPDATESTEP_H__

#include <local_ministep.h>

typedef struct local_updatestep_struct local_updatestep_type;

local_updatestep_type * local_updatestep_alloc( const char * name );
void                    local_updatestep_free__(void * arg);
void                    local_updatestep_add_ministep( local_updatestep_type * updatestep , local_ministep_type * ministep);
local_ministep_type   * local_updatestep_iget_ministep( const local_updatestep_type * updatestep , int index);
local_updatestep_type * local_updatestep_alloc_copy( const local_updatestep_type * src , const char * name );
int                     local_updatestep_get_num_ministep( const local_updatestep_type * updatestep );

#endif
