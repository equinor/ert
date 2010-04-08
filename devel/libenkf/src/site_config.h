#ifndef __SITE_CONFIG_H__
#define __SITE_CONFIG_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <config.h>
#include <ext_joblist.h>
#include <stringlist.h>
#include <forward_model.h>
#include <stdbool.h>

typedef struct site_config_struct site_config_type;

bool                     site_config_get_statoil_mode(const site_config_type * site_config );
void                     site_config_update_lsf_request(site_config_type *  , const forward_model_type *);
site_config_type       * site_config_alloc(const config_type * , bool *);
void                     site_config_free(site_config_type *); 
ext_joblist_type       * site_config_get_installed_jobs( const site_config_type * );
job_queue_type         * site_config_get_job_queue( const site_config_type * );
void                     site_config_set_ens_size( site_config_type * site_config , int ens_size );

void                     site_config_set_max_running_lsf( site_config_type * site_config , int max_running_lsf);
int                      site_config_get_max_running_lsf( const site_config_type * site_config );
void                     site_config_set_max_running_rsh( site_config_type * site_config , int max_running_rsh);
int                      site_config_get_max_running_rsh( const site_config_type * site_config);
void                     site_config_set_max_running_local( site_config_type * site_config , int max_running_local);
int                      site_config_get_max_running_local( const site_config_type * site_config );
void                     site_config_setenv( site_config_type * site_config , const char * variable, const char * value);
hash_type              * site_config_get_env_hash( const site_config_type * site_config );
void                     site_config_clear_env( site_config_type * site_config );
void                     site_config_clear_pathvar( site_config_type * site_config );
stringlist_type        * site_config_get_path_variables( const site_config_type * site_config );
stringlist_type        * site_config_get_path_values( const site_config_type * site_config );


#ifdef __cplusplus
}
#endif
#endif 
