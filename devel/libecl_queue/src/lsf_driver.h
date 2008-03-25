#ifndef __LSF_DRIVER_H__
#define __LSF_DRIVER_H__
#include <basic_queue_driver.h>

typedef struct lsf_driver_struct lsf_driver_type;
typedef struct lsf_job_struct    lsf_job_type;


void 	  * lsf_driver_alloc(const char * , const char * );
void 	    lsf_driver_free(lsf_driver_type * );
void        lsf_driver_free__(basic_queue_driver_type * );

#endif 
