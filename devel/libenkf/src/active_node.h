#ifndef __ACTIVE_NODE_H__
#define __ACTIVE_NODE_H__


typedef struct active_var_struct active_var_type;
typedef struct active_obs_struct active_obs_type;


typedef void (active_config_destructor_ftype) (void *);  /* A destructor for the (node-spesific) object holding inactive/active information. */

#endif
