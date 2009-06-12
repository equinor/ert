#ifndef __LOCAL_REPORTSTEP_H__
#define __LOCAL_REPORTSTEP_H__

#include <local_ministep.h>

typedef struct local_reportstep_struct local_reportstep_type;

local_reportstep_type * local_reportstep_alloc( const char * name );
void                    local_reportstep_free__(void * arg);
void                    local_reportstep_add_ministep( local_reportstep_type * reportstep , local_ministep_type * ministep);
local_ministep_type   * local_reportstep_iget_ministep( const local_reportstep_type * reportstep , int index);
local_reportstep_type * local_reportstep_alloc_copy( const local_reportstep_type * src , const char * name );
int                     local_reportstep_get_num_ministep( const local_reportstep_type * reportstep);

#endif
