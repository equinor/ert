#ifndef __RSH_DRIVER_H__
#define __RSH_DRIVER_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <hash.h>

#define RSH_HOST           "RSH_HOST"
#define RSH_HOSTLIST       "RSH_HOSTLIST"
#define RSH_CMD            "RSH_CMD" 
#define RSH_CLEAR_HOSTLIST "RSH_CLEAR_HOSTLIST"

typedef struct rsh_driver_struct rsh_driver_type;
typedef struct rsh_job_struct    rsh_job_type;

void        rsh_driver_add_host(rsh_driver_type * , const char * , int );
void      * rsh_driver_alloc( );

void * rsh_driver_submit_job(void * __driver , 
                             const char  * submit_cmd     , 
                             const char  * run_path       , 
                             const char  * job_name ,
                             int           argc,     
                             const char ** argv );
void            rsh_driver_kill_job(void * __driver , void * __job);
void            rsh_driver_free__(void * __driver );
job_status_type rsh_driver_get_job_status(void * __driver , void * __job);
void            rsh_driver_free_job(void * __job);


void         rsh_driver_set_option( void * __driver, const char * option_key , const void * value );
const void * rsh_driver_get_option( const void * __driver , const char * option_key);

#ifdef __cplusplus
}
#endif
#endif 
