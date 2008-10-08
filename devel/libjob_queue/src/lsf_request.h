#ifndef __LSF_REQUEST_H__
#define __LSF_REQUEST_H__


typedef struct lsf_request_struct lsf_request_type;


lsf_request_type * lsf_request_alloc();
void               lsf_request_free(lsf_request_type *);
void               lsf_request_reset(lsf_request_type *);   
void               lsf_request_update(lsf_request_type * , const char *);
const char *       lsf_request_get(const lsf_request_type *);

#endif
