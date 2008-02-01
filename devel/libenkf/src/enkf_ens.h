#ifndef __ENKF_ENS_H__
#define __ENKF_ENS_H__
#include <stdbool.h>
#include <enkf_config_node.h>
#include <enkf_types.h>
#include <obs_data.h>
#include <path_fmt.h>
#include <enkf_fs.h>
#include <meas_vector.h>
#include <sched_file.h>

/*
#define LOG_INPUT    1
#define LOG_ENKF     2 
#define LOG_OUTPUT   4
*/

/*****************************************************************/



typedef struct enkf_ens_struct enkf_ens_type;

void 			      enkf_ens_set_state_run_path(const enkf_ens_type * , int );
void 			      enkf_ens_set_state_eclbase(const enkf_ens_type * , int );

const char            	   ** enkf_ens_get_well_list_ref(const enkf_ens_type * , int *);
bool                  	      enkf_ens_get_endian_swap(const enkf_ens_type * );
bool                          enkf_ens_get_fmt_file(const enkf_ens_type * );
enkf_ens_type         	    * enkf_ens_alloc(int , enkf_fs_type * , const char * , const char * , sched_file_type * , bool , bool);
enkf_impl_type        	      enkf_ens_impl_type(const enkf_ens_type *, const char * );
bool                  	      enkf_ens_has_key(const enkf_ens_type * , const char *);
void                  	      enkf_ens_add_type(enkf_ens_type * , const char * , enkf_var_type , enkf_impl_type , const void *);
void                  	      enkf_ens_add_type0(enkf_ens_type * , const char * , int , enkf_var_type , enkf_impl_type );
void                  	      enkf_ens_add_well(enkf_ens_type * , const char * , int , const char ** );
const enkf_config_node_type * enkf_ens_get_config_ref(const enkf_ens_type * , const char * );
void                          enkf_ens_free(enkf_ens_type * );
enkf_fs_type                * enkf_ens_get_fs_ref(const enkf_ens_type *);
meas_vector_type            * enkf_ens_iget_meas_vector(const enkf_ens_type * , int );

#endif
