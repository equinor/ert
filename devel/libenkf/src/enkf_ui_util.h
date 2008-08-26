#ifndef __ENKF_UI_UTIL_H__
#define __ENKF_UI_UTIL_H__


#include <enkf_config.h>
#include <enkf_types.h>
#include <field_config.h>
#include <enkf_sched.h>
#include <enkf_config.h>


void        enkf_ui_util_scanf_parameter(const enkf_config_type *  , int , bool , char ** , int *  , state_enum * , int *);
state_enum  enkf_ui_util_scanf_state(const char * , int , bool);
int         enkf_ui_util_scanf_ijk(const field_config_type * , int);
void        enkf_ui_util_scanf_ijk__(const field_config_type * , int  , int * , int * , int *);
bool      * enkf_ui_util_scanf_alloc_report_active(const enkf_sched_type * , int );
bool      * enkf_ui_util_scanf_alloc_iens_active(const enkf_config_type * , int , int * , int *);


#endif
