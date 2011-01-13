#ifndef __LSF_DRIVER_H__
#define __LSF_DRIVER_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <queue_driver.h>
#include <stringlist.h>


#define LSF_QUEUE     1
#define LSF_RESOURCE  2
#define LSF_SERVER    3
#define LSF_NUM_CPU   4
  



typedef struct lsf_driver_struct lsf_driver_type;
typedef struct lsf_job_struct    lsf_job_type;


void      * lsf_driver_alloc( );
int         lsf_driver_get_num_cpu( const void * __lsf_driver );
void        lsf_driver_set_num_cpu( void * __lsf_driver , int num_cpu);
void        lsf_driver_set_queue_name( lsf_driver_type * driver, const char * queue_name );
void        lsf_driver_set_resource_request( lsf_driver_type * driver, const char * resource_request );
void        lsf_driver_set_remote_server( lsf_driver_type * driver, const char * remote_lsf_server);

void * lsf_driver_submit_job(void * __driver , 
                             const char  * submit_cmd     , 
                             const char  * run_path       , 
                             const char  * job_name ,
                             int           argc,     
                             const char ** argv );
void            lsf_driver_kill_job(void * __driver , void * __job);
void            lsf_driver_free__(void * __driver );
job_status_type lsf_driver_get_job_status(void * __driver , void * __job);
void            lsf_driver_free_job(void * __job);
void            lsf_driver_display_info( void * __driver , void * __job);


bool            lsf_driver_has_option( const void * __driver , int option_id);
const  void   * lsf_driver_get_option( const void * __driver , int option_id);
void            lsf_driver_set_option( void * __driver , int option_id , const void * value);


UTIL_SAFE_CAST_HEADER( lsf_driver );


#ifdef __cplusplus
}
#endif
#endif 
