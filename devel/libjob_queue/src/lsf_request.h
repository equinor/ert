#ifndef __LSF_REQUEST_H__
#define __LSF_REQUEST_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <ext_joblist.h>
#include <stringlist.h>
#include <stdbool.h>

typedef struct lsf_request_struct lsf_request_type;


lsf_request_type * lsf_request_alloc(bool statoil_mode);
void               lsf_request_free(lsf_request_type *);
void               lsf_request_reset(lsf_request_type *);   
void               lsf_request_update(lsf_request_type * , const ext_job_type * , bool);
const char *       lsf_request_get(const lsf_request_type *);
void               lsf_request_add_manual_request(lsf_request_type *  , const char * );

#ifdef __cplusplus
}
#endif
#endif
