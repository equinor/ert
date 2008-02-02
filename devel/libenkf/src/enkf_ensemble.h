#ifndef __ENKF_ENSEMBLE_H__
#define __ENKF_ENSEMBLE_H__
#include <stdbool.h>
#include <enkf_config_node.h>
#include <enkf_types.h>
#include <obs_data.h>
#include <path_fmt.h>
#include <enkf_fs.h>
#include <sched_file.h>


/*****************************************************************/



typedef struct enkf_ensemble_struct enkf_ensemble_type;

void 			      enkf_ensemble_set_state_run_path(const enkf_ensemble_type * , int );
void 			      enkf_ensemble_set_state_eclbase(const enkf_ensemble_type * , int );
const char                  * enkf_ensemble_get_data_file(const enkf_ensemble_type * );
const char            	   ** enkf_ensemble_get_well_list_ref(const enkf_ensemble_type * , int *);
bool                  	      enkf_ensemble_get_endian_swap(const enkf_ensemble_type * );
bool                          enkf_ensemble_get_fmt_file(const enkf_ensemble_type * );
enkf_ensemble_type         	    * enkf_ensemble_alloc(int , enkf_fs_type * , const char * , const char * , const char * , sched_file_type * , bool , bool , bool);
enkf_impl_type        	      enkf_ensemble_impl_type(const enkf_ensemble_type *, const char * );
bool                  	      enkf_ensemble_has_key(const enkf_ensemble_type * , const char *);
void                  	      enkf_ensemble_add_type(enkf_ensemble_type * , const char * , enkf_var_type , enkf_impl_type , const char * , const void *);
void                  	      enkf_ensemble_add_type0(enkf_ensemble_type * , const char * , int , enkf_var_type , enkf_impl_type );
void                  	      enkf_ensemble_add_well(enkf_ensemble_type * , const char * , int , const char ** );
void                          enkf_ensemble_add_gen_kw(enkf_ensemble_type * , const char * );
const enkf_config_node_type * enkf_ensemble_get_config_ref(const enkf_ensemble_type * , const char * );
void                          enkf_ensemble_free(enkf_ensemble_type * );
enkf_fs_type                * enkf_ensemble_get_fs_ref(const enkf_ensemble_type *);
void                          enkf_ensemble_set_data_kw(enkf_ensemble_type * , const char * , const char *);
void                          enkf_ensemble_add_data_kw(enkf_ensemble_type * , const char * , const char *);
void                          enkf_ensemble_init_eclipse(enkf_ensemble_type * );

void 			      enkf_ensemble_load_ecl_init_mt(enkf_ensemble_type * enkf_ensemble , int );
void 			      enkf_ensemble_iload_ecl_mt(enkf_ensemble_type *enkf_ensemble , int );
void 			      enkf_ensemble_load_ecl_complete_mt(enkf_ensemble_type *);
void                          enkf_ensemble_analysis(enkf_ensemble_type * );

#endif
