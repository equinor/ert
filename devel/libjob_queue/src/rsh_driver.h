#ifndef __RSH_DRIVER_H__
#define __RSH_DRIVER_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stringlist.h>

typedef struct rsh_driver_struct rsh_driver_type;
typedef struct rsh_job_struct    rsh_job_type;

void        rsh_driver_add_host(rsh_driver_type * , const char * , int );
void 	  * rsh_driver_alloc(const char *, const stringlist_type *);

#ifdef __cplusplus
}
#endif
#endif 
