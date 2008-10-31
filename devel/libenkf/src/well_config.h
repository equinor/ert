#ifndef __WELL_CONFIG_H__
#define __WELL_CONFIG_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdlib.h>
#include <enkf_macros.h>



typedef struct well_config_struct well_config_type;

void               well_config_set_obs_config_file(well_config_type * , const char * );
const char       * well_config_get_config_txt_file_ref(const well_config_type * );
well_config_type * well_config_fscanf_alloc(const char * , const char * );
well_config_type * well_config_alloc(const char * , int , const char ** );
void               well_config_free(well_config_type * );
int                well_config_get_active_mask(const well_config_type *);
const char       * well_config_get_name(const well_config_type * );
int                well_config_get_var_index(const well_config_type * , const char * );
const char      ** well_config_get_var_list_ref(const well_config_type *);
void               well_config_add_var(well_config_type *  , const char * );
bool               well_config_has_var(const well_config_type * , const char * );
void               well_config_summarize(const well_config_type * );


GET_ACTIVE_SIZE_HEADER(well);
GET_ACTIVE_LIST_HEADER(well);
GET_DATA_SIZE_HEADER(well);
VOID_CONFIG_FREE_HEADER(well);
#ifdef __cplusplus
}
#endif
#endif
