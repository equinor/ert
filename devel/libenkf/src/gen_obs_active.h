#ifndef __GEN_OBS_ACTIVE_H__
#define __GEN_OBS_ACTIVE_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <enkf_macros.h>

typedef struct gen_obs_active_struct gen_obs_active_type;

int  	    gen_obs_active_get_active_size(const gen_obs_active_type * );
int  	  * gen_obs_active_alloc_list_copy(const gen_obs_active_type * );
const int * gen_obs_active_get_active_list(const gen_obs_active_type * );

SAFE_CAST_HEADER(gen_obs_active);


#ifdef __cplusplus
}
#endif
#endif
