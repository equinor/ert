#ifndef __LOCAL_DRIVER_H__
#define __LOCAL_DRIVER_H__

typedef struct local_driver_struct local_driver_type;
typedef struct local_job_struct    local_job_type;


void 	  * local_driver_alloc();
void 	    local_driver_free(local_driver_type * );
void        local_driver_free__(basic_queue_driver_type * );

#endif 
