#ifndef __WELL_CONFIG_H__
#define __WELL_CONFIG_H__

#include <stdbool.h>
#include <stdlib.h>
#include <enkf_macros.h>

struct well_config_struct {
  CONFIG_STD_FIELDS;
  char *  well_name;
  char ** var_list;
};



typedef struct well_config_struct well_config_type;

const char       * well_config_get_config_txt_file_ref(const well_config_type * );
well_config_type * well_config_fscanf_alloc(const char * , const char * );
well_config_type * well_config_alloc(const char * , int , const char ** );
void               well_config_free(well_config_type * );
int                well_config_get_active_mask(const well_config_type *);
const char       * well_config_get_well_name_ref(const well_config_type * );
int                well_config_get_var_index(const well_config_type * , const char * );
const char      ** well_config_get_var_list_ref(const well_config_type *);
void               well_config_add_var(well_config_type *  , const char * );
bool               well_config_has_var(const well_config_type * , const char * );



GET_DATA_SIZE_HEADER(well);
VOID_CONFIG_FREE_HEADER(well);
#endif
