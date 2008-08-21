#ifndef __ENKF_UI_UTIL_H__
#define __ENKF_UI_UTIL_H__


#include <enkf_config.h>
#include <enkf_types.h>
#include <field_config.h>

void        enkf_ui_util_scanf_parameter(const enkf_config_type *  , char ** , int *  , state_enum * , int *);
state_enum  enkf_ui_util_scanf_state(const char * );
int         enkf_ui_util_scanf_ijk(const field_config_type * );

#endif
