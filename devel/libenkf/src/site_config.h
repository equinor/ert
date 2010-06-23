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

const char            *  site_config_get_license_root_path__( const site_config_type * site_config );
void                     site_config_set_license_root_path( site_config_type * site_config , const char * license_root_path);


void                     site_config_set_num_cpu( site_config_type * site_config , int num_cpu );
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

void                     site_config_clear_rsh_host_list( site_config_type * site_config );
hash_type              * site_config_get_rsh_host_list( const site_config_type * site_config );
void                     site_config_add_rsh_host( const site_config_type * site_config , const char * rsh_host , int max_running);

void                     site_config_set_lsf_queue( site_config_type * site_config , const char * lsf_queue);
const char             * site_config_get_lsf_queue( const site_config_type * site_config );
void                     site_config_set_lsf_request( site_config_type * site_config , const char * lsf_request);
const char             * site_config_get_lsf_request( const site_config_type * site_config );

const char             * site_config_get_job_queue_name( const site_config_type * site_config );
void                     site_config_set_job_queue( site_config_type * site_config , const char * queue_name );

void                     site_config_set_rsh_command( site_config_type * site_config , const char * rsh_command);
const char             * site_config_get_rsh_command( const site_config_type * site_config );


void                     site_config_set_job_script( site_config_type * site_config , const char * job_script );
const char             * site_config_get_job_script( const site_config_type * site_config );

void                     site_config_set_max_submit( site_config_type * site_config , int max_submit );
int                      site_config_get_max_submit(const site_config_type * site_config );

bool                     site_config_queue_is_running( const site_config_type * site_config );
int                      site_config_install_job(site_config_type * site_config , const char * job_name , const char * install_file);
bool                     site_config_del_job( site_config_type * site_config , const char * job_name );

void                     site_config_set_umask( site_config_type * site_config , mode_t umask);
mode_t                   site_config_get_umask( const site_config_type * site_config );



#ifdef __cplusplus
}
#endif
#endif 
