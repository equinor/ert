#ifndef __ENKF_TUI_UTIL_H__
#define __ENKF_TUI_UTIL_H__


#include <enkf_types.h>
#include <field_config.h>
#include <enkf_fs.h>
#include <enkf_config_node.h>
#include <ensemble_config.h>


void                          enkf_tui_util_scanf_report_steps(int  , int  , int *  , int * );
const enkf_config_node_type * enkf_tui_util_scanf_key(const ensemble_config_type *  , int , enkf_impl_type ,  enkf_var_type);
state_enum		      enkf_tui_util_scanf_state(const char * , int , bool);
int       		      enkf_tui_util_scanf_ijk(const field_config_type * , int);
void      		      enkf_tui_util_scanf_ijk__(const field_config_type * , int  , int * , int * , int *);
bool      		    * enkf_tui_util_scanf_alloc_report_active(int , int );
bool      		    * enkf_tui_util_scanf_alloc_iens_active(int , int , int * , int *);
void      		      enkf_tui_util_get_time(enkf_fs_type * , const enkf_config_node_type * , enkf_node_type * , state_enum , int  , int  , int  , int  , double *  , double *  );
void                          enkf_tui_util_scanf_iens_range(const char * , int  , int  , int *  , int * );
int                           enkf_tui_util_scanf_report_step(int , const char *  , int );
void                          enkf_tui_util_msg(const char * , ...);
int                           enkf_tui_util_scanf_int_with_default(const char * prompt , int prompt_len , bool * default_used);
#endif
