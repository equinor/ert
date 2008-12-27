#ifndef __ENKF_ENSEMBLE_H__
#define __ENKF_ENSEMBLE_H__
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

/*****************************************************************/



typedef struct enkf_main_struct enkf_main_type;
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
void                          enkf_main_load_ecl_init_mt(enkf_main_type * enkf_main , int );
void                          enkf_main_load_ecl_complete_mt(enkf_main_type *);
void                          enkf_main_iload_ecl_mt(enkf_main_type *enkf_main , int );
void                          enkf_main_run(enkf_main_type * , const bool * ,  int  , state_enum );
void                          enkf_main_run_step(enkf_main_type *, run_mode_type , const bool * , int, state_enum , int , int, bool, const stringlist_type *);
void                          enkf_main_set_data_kw(enkf_main_type * , const char * , const char *);
void                          enkf_main_set_state_run_path(const enkf_main_type * , int );
void                          enkf_main_set_state_eclbase(const enkf_main_type * , int );
enkf_main_type              * enkf_main_safe_cast( void * );
void                          enkf_main_interactive_set_runpath__(void * );
enkf_main_type              * enkf_main_bootstrap(const char * , const char * );

lock_mode_type                enkf_main_get_runlock_mode(const enkf_main_type * );
enkf_fs_type                * enkf_main_get_fs_ref(const enkf_main_type *);
enkf_impl_type                enkf_main_impl_type(const enkf_main_type *, const char * );
enkf_state_type             * enkf_main_iget_state(const enkf_main_type * , int );

const enkf_config_node_type * enkf_main_get_config_node(const enkf_main_type * , const char *);
const sched_file_type       * enkf_main_get_sched_file(const enkf_main_type *);
const ensemble_config_type  * enkf_main_get_ensemble_config(const enkf_main_type * );
const enkf_sched_type       * enkf_main_get_enkf_sched(const enkf_main_type *);
      model_config_type     * enkf_main_get_model_config( const enkf_main_type * );
enkf_fs_type                * enkf_main_get_fs(const enkf_main_type * );
enkf_obs_type               * enkf_main_get_obs(const enkf_main_type * );

void       * enkf_main_get_enkf_config_node_type(const ensemble_config_type *, const char *);
void 	     enkf_main_set_field_config_iactive(const ensemble_config_type *, int);
const char * enkf_main_get_image_viewer(const enkf_main_type * );

void         enkf_main_analysis_update(enkf_main_type * , int );

#ifdef __cplusplus
}
#endif
#endif
