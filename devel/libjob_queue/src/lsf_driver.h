#ifndef __LSF_DRIVER_H__
#define __LSF_DRIVER_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <basic_queue_driver.h>
#include <stringlist.h>





typedef struct lsf_driver_struct lsf_driver_type;
typedef struct lsf_job_struct    lsf_job_type;


void 	  * lsf_driver_alloc(const char * , int num_cpu);
int         lsf_driver_get_num_cpu( const void * __lsf_driver );
void        lsf_driver_set_num_cpu( void * __lsf_driver , int num_cpu);




#ifdef __cplusplus
}
#endif
#endif 
