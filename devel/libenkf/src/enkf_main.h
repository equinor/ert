#ifndef __ENKF_MAIN_H__
#define __ENKF_MAIN_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdbool.h>
#include <enkf_config_node.h>
#include <enkf_types.h>
#include <enkf_state.h>
#include <obs_data.h>
#include <path_fmt.h>
#include <enkf_fs.h>
#include <sched_file.h>
#include <job_queue.h>
#include <ext_joblist.h>
#include <stringlist.h>
#include <enkf_obs.h>
#include <forward_model.h>
#include <misfit_table.h>
#include <plot_config.h>
#include <set.h>
#include <member_config.h>
#include <analysis_config.h>
  
/*****************************************************************/



typedef struct enkf_main_struct enkf_main_type;
member_config_type          * enkf_main_iget_member_config(const enkf_main_type * enkf_main , int iens);
misfit_table_type           * enkf_main_get_misfit(const enkf_main_type * enkf_main);
void                          enkf_main_del_unused_static(enkf_main_type * , int );
const char                  * enkf_main_get_data_file(const enkf_main_type * );
const char                 ** enkf_main_get_well_list_ref(const enkf_main_type * , int *);

bool                          enkf_main_get_endian_swap(const enkf_main_type * );
bool                          enkf_main_get_fmt_file(const enkf_main_type * );
bool                          enkf_main_has_key(const enkf_main_type * , const char *);

void                          enkf_main_add_gen_kw(enkf_main_type * , const char * );
void                          enkf_main_add_type(enkf_main_type * , const char * , enkf_var_type , enkf_impl_type , const char * , const void *);
void                          enkf_main_add_type0(enkf_main_type * , const char * , int , enkf_var_type , enkf_impl_type );
void                          enkf_main_add_well(enkf_main_type * , const char * , int , const char ** );
void                          enkf_main_analysis(enkf_main_type * );
void                          enkf_main_free(enkf_main_type * );
void                          enkf_main_init_eclipse(enkf_main_type * , int , int );
void                          enkf_main_init_run( enkf_main_type * enkf_main, run_mode_type run_mode);
void                          enkf_main_load_ecl_init_mt(enkf_main_type * enkf_main , int );
void                          enkf_main_load_ecl_complete_mt(enkf_main_type *);
void                          enkf_main_iload_ecl_mt(enkf_main_type *enkf_main , int );
void                          enkf_main_run(enkf_main_type * , run_mode_type , const bool * ,  int  , int , state_enum );
  //void                          enkf_main_run_step(enkf_main_type *, run_mode_type , const bool * , int, state_enum , int , int, bool, forward_model_type *);
void                          enkf_main_set_data_kw(enkf_main_type * , const char * , const char *);
void                          enkf_main_set_state_run_path(const enkf_main_type * , int );
void                          enkf_main_set_state_eclbase(const enkf_main_type * , int );
void                          enkf_main_interactive_set_runpath__(void * );
enkf_main_type              * enkf_main_bootstrap(const char * , const char * );

enkf_node_type             ** enkf_main_get_node_ensemble(const enkf_main_type * enkf_main , const char * key , int report_step , state_enum load_state);
void                          enkf_main_node_mean( const enkf_node_type ** ensemble , int ens_size , enkf_node_type * mean );
void                          enkf_main_node_std( const enkf_node_type ** ensemble , int ens_size , const enkf_node_type * mean , enkf_node_type * std);

enkf_fs_type                * enkf_main_get_fs_ref(const enkf_main_type *);
enkf_impl_type                enkf_main_impl_type(const enkf_main_type *, const char * );
enkf_state_type             * enkf_main_iget_state(const enkf_main_type * , int );

const enkf_config_node_type * enkf_main_get_config_node(const enkf_main_type * , const char *);
const sched_file_type       * enkf_main_get_sched_file(const enkf_main_type *);
ecl_config_type             * enkf_main_get_ecl_config(const enkf_main_type * enkf_main);
ensemble_config_type        * enkf_main_get_ensemble_config(const enkf_main_type * enkf_main);
int                           enkf_main_get_ensemble_size( const enkf_main_type * enkf_main );
int   			      enkf_main_get_history_length( const enkf_main_type * );
bool  			      enkf_main_has_prediction( const enkf_main_type *  );
//const enkf_sched_type       * enkf_main_get_enkf_sched(const enkf_main_type *);
model_config_type           * enkf_main_get_model_config( const enkf_main_type * );
plot_config_type            * enkf_main_get_plot_config( const enkf_main_type * enkf_main );
enkf_fs_type                * enkf_main_get_fs(const enkf_main_type * );
enkf_obs_type               * enkf_main_get_obs(const enkf_main_type * );
analysis_config_type        * enkf_main_get_analysis_config(const enkf_main_type * );

void       * enkf_main_get_enkf_config_node_type(const ensemble_config_type *, const char *);
void 	     enkf_main_set_field_config_iactive(const ensemble_config_type *, int);
const char * enkf_main_get_image_viewer(const enkf_main_type * );
const char * enkf_main_get_plot_driver(const enkf_main_type * enkf_main );
void         enkf_main_analysis_update(enkf_main_type * , int , int );
const char * enkf_main_get_image_type(const enkf_main_type * enkf_main);
void         enkf_main_UPDATE(enkf_main_type * enkf_main , bool merge_observations , int step1 , int step2);
void         enkf_main_initialize(enkf_main_type * enkf_main , const stringlist_type * param_list , int iens1 , int iens2);
void                enkf_main_set_misfit_table( enkf_main_type * enkf_main , misfit_table_type * misfit);
misfit_table_type * enkf_main_get_misfit_table( const enkf_main_type * enkf_main );

void                     enkf_main_store_pid(const char * argv0);
void                     enkf_main_delete_pid( );
void                     enkf_main_list_users(  set_type * users , const char * executable );
matrix_type      *       enkf_main_getA(enkf_main_type * enkf_main , const local_ministep_type * ministep, int report_step , hash_type * use_count);
const ext_joblist_type * enkf_main_get_installed_jobs( const enkf_main_type * enkf_main );
SAFE_CAST_HEADER(enkf_main)


/*****************************************************************/

int                      enkf_main_hello( const enkf_main_type * enkf_main );
#ifdef __cplusplus
}
#endif
#endif
