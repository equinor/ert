#ifndef __LSF_DRIVER_H__
#define __LSF_DRIVER_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <basic_queue_driver.h>
#include <stringlist.h>

typedef struct lsf_driver_struct lsf_driver_type;
typedef struct lsf_job_struct    lsf_job_type;


void 	  * lsf_driver_alloc(const char *);
void 	    lsf_driver_free(lsf_driver_type * );
void        lsf_driver_free__(basic_queue_driver_type * );

#ifdef __cplusplus
}
#endif
#endif 
