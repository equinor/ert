#ifndef __LOCAL_DRIVER_H__
#define __LOCAL_DRIVER_H__
#ifdef __cplusplus
extern "C" {
#endif

typedef struct local_driver_struct local_driver_type;
typedef struct local_job_struct    local_job_type;


void      * local_driver_alloc();


void * local_driver_submit_job(void * __driver , 
                             const char  * submit_cmd     , 
                             const char  * run_path       , 
                             const char  * job_name ,
                             int           argc,     
                             const char ** argv );
void            local_driver_kill_job(void * __driver , void * __job);
void            local_driver_free__(void * __driver );
job_status_type local_driver_get_job_status(void * __driver , void * __job);
void            local_driver_free_job(void * __job);





#ifdef __cplusplus
}
#endif
#endif 
