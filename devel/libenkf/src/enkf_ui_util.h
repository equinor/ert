#ifndef __ENKF_UI_UTIL_H__
#define __ENKF_UI_UTIL_H__


#include <enkf_types.h>
#include <field_config.h>
#include <enkf_sched.h>
#include <enkf_fs.h>
#include <enkf_config_node.h>
#include <ensemble_config.h>


const enkf_config_node_type * enkf_ui_util_scanf_parameter(const ensemble_config_type *  , int , bool , enkf_impl_type ,  enkf_var_type , int *  , state_enum * , int *);
state_enum		      enkf_ui_util_scanf_state(const char * , int , bool);
int       		      enkf_ui_util_scanf_ijk(const field_config_type * , int);
void      		      enkf_ui_util_scanf_ijk__(const field_config_type * , int  , int * , int * , int *);
bool      		    * enkf_ui_util_scanf_alloc_report_active(const enkf_sched_type * , int );
bool      		    * enkf_ui_util_scanf_alloc_iens_active(int , int , int * , int *);
void      		      enkf_ui_util_get_time(enkf_fs_type * , const enkf_config_node_type * , enkf_node_type * , state_enum , int  , int  , int  , int  , double *  , double *  );
void                          enkf_ui_util_scanf_iens_range(int  , int  , int *  , int * );

#endif
