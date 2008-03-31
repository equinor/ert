#ifndef __RSH_DRIVER_H__
#define __RSH_DRIVER_H__

typedef struct rsh_driver_struct rsh_driver_type;
typedef struct rsh_job_struct    rsh_job_type;


void 	  * rsh_driver_alloc(const char *);
void 	    rsh_driver_free(rsh_driver_type * );
void        rsh_driver_free__(basic_queue_driver_type * );

#endif 
