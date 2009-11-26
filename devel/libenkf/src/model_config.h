#ifndef __MODEL_CONFIG_H__
#define __MODEL_CONFIG_H__
#ifdef __cplusplus 
extern "C" {
#endif


#include <time.h>
#include <config.h>
#include <ext_joblist.h>
#include <enkf_sched.h>
#include <history.h>
#include <sched_file.h>
#include <path_fmt.h>
#include <forward_model.h>
#include <enkf_types.h>
#include <sched_file.h>
#include <stdbool.h>
#include <fs_types.h>

typedef struct model_config_struct model_config_type;
void                  model_config_set_enspath( model_config_type * model_config , const char * enspath);
void                  model_config_set_dbase_type( model_config_type * model_config , const char * dbase_type_string);
const char          * model_config_get_enspath( const model_config_type * model_config);
fs_driver_impl        model_config_get_dbase_type(const model_config_type * model_config );
void                  model_config_init_internalization( model_config_type * );
void                  model_config_set_internalize_state( model_config_type *  , int );
void                  model_config_set_internalize_results( model_config_type *  , int ) ;
void                  model_config_set_load_state( model_config_type *  , int );
void                  model_config_set_load_results( model_config_type *  , int ) ;
int 		      model_config_get_history_length(const model_config_type * );
bool                  model_config_has_prediction(const model_config_type * );
int                   model_config_get_last_history_restart(const model_config_type * );
model_config_type   * model_config_alloc(const config_type * , int ens_size , const ext_joblist_type * , int , const sched_file_type * , bool , bool);
void                  model_config_free(model_config_type *);
path_fmt_type       * model_config_get_runpath_fmt(const model_config_type * );
enkf_sched_type     * model_config_get_enkf_sched(const model_config_type * );
history_type        * model_config_get_history(const model_config_type * );
void                  model_config_interactive_set_runpath__(void * arg);
forward_model_type  * model_config_get_std_forward_model( const model_config_type * );
bool                  model_config_internalize_state( const model_config_type *, int );
bool                  model_config_internalize_results( const model_config_type *, int );
bool                  model_config_load_state( const model_config_type *, int );
bool                  model_config_load_results( const model_config_type *, int );
void                  model_config_set_enkf_sched(model_config_type *  , const ext_joblist_type * , run_mode_type , bool);
bool                  model_config_resample_when_fail( const model_config_type * config );
int                   model_config_get_max_internal_submit( const model_config_type * config );
const char          * model_config_iget_casename( const model_config_type * model_config , int index);

#ifdef __cplusplus 
}
#endif
#endif


