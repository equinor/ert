#ifndef __LOCAL_CONFIG_H__
#define __LOCAL_CONFIG_H__

#include <local_updatestep.h>
#include <local_ministep.h>

typedef struct local_config_struct local_config_type;

local_config_type     	    * local_config_alloc( int history_length );
void                  	      local_config_free( local_config_type * local_config );

local_updatestep_type 	    * local_config_alloc_updatestep( local_config_type * local_config , const char * key );
local_ministep_type   	    * local_config_alloc_ministep( local_config_type * local_config , const char * key );
local_ministep_type         * local_config_alloc_ministep_copy( local_config_type * local_config , const char * src_key , const char * new_key);
void                  	      local_config_set_default_updatestep( local_config_type * local_config , const char * default_key);
const local_updatestep_type * local_config_iget_updatestep( const local_config_type * local_config , int index);


#endif
