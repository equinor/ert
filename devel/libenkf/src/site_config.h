#ifndef __SITE_CONFIG_H__
#define __SITE_CONFIG_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <config.h>
#include <ext_joblist.h>
#include <stringlist.h>
#include <forward_model.h>

typedef struct site_config_struct site_config_type;

const char             * site_config_get_image_type(site_config_type * site_config);
const char             * site_config_get_image_viewer(site_config_type * site_config);
void                     site_config_update_lsf_request(site_config_type *  , const forward_model_type *);
site_config_type       * site_config_alloc(const config_type * , int , bool *);
void                     site_config_free(site_config_type *); 
ext_joblist_type       * site_config_get_installed_jobs( const site_config_type * );
job_queue_type         * site_config_get_job_queue( const site_config_type * );

#ifdef __cplusplus
}
#endif
#endif 
