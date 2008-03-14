#ifndef __LSF_DRIVER_H__
#define __LSF_DRIVER_H__

typedef struct lsf_driver_struct lsf_driver_type;
typedef struct lsf_job_struct    lsf_job_type;


void 	  * lsf_driver_alloc(const char * , const char * );
void 	    lsf_driver_free(lsf_driver_type * );


#endif 
