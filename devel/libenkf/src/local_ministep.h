#ifndef __LOCAL_MINISTEP_H__
#define __LOCAL_MINISTEP_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <active_list.h>
#include <hash.h>
#include <stringlist.h>
#include <util.h>

typedef struct local_ministep_struct local_ministep_type;

local_ministep_type * local_ministep_alloc(const char * name);
void 		      local_ministep_free(local_ministep_type * ministep);
void 		      local_ministep_free__(void * arg);
void 		      local_ministep_add_obs(local_ministep_type * ministep, const char * obs_key);
void 		      local_ministep_add_node(local_ministep_type * ministep, const char *node_key);
active_list_type    * local_ministep_get_obs_active_list(const local_ministep_type * ministep , const char * obs_key );
active_list_type    * local_ministep_get_node_active_list(const local_ministep_type * ministep , const char * node_key );
hash_iter_type 	    * local_ministep_alloc_node_iter(const local_ministep_type * ministep);
hash_iter_type 	    * local_ministep_alloc_obs_iter(const local_ministep_type * ministep);
stringlist_type     * local_ministep_alloc_node_keys(const local_ministep_type * ministep);
local_ministep_type * local_ministep_alloc_copy( const local_ministep_type * src , const char * name);
void 		      local_ministep_del_obs( local_ministep_type * ministep , const char * obs_key);
void 		      local_ministep_del_node( local_ministep_type * ministep , const char * node_key);
const char          * local_ministep_get_name( const local_ministep_type * ministep );
void                  local_ministep_clear_nodes( local_ministep_type * ministep);
void                  local_ministep_clear_observations( local_ministep_type * ministep);


UTIL_SAFE_CAST_HEADER(local_ministep);
UTIL_IS_INSTANCE_HEADER(local_ministep);

#ifdef __cplusplus
}
#endif
#endif
