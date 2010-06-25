#ifndef __FORWARD_MODEL_H__
#define __FORWARD_MODEL_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <subst_list.h>
#include <ext_joblist.h>
#include <stdbool.h>
#include <stringlist.h>

typedef struct  forward_model_struct forward_model_type ;


stringlist_type        * forward_model_alloc_joblist( const forward_model_type * forward_model );
const ext_joblist_type * forward_model_get_joblist(const forward_model_type * );
void                     forward_model_clear( forward_model_type * forward_model );
void                   	 forward_model_fprintf(const forward_model_type *  , FILE * );
forward_model_type     * forward_model_alloc(const ext_joblist_type * ext_joblist, const char * lsf_request);
void                     forward_model_parse_init(forward_model_type * forward_model , const char * input_string );
void 	               	 forward_model_python_fprintf(const forward_model_type *  , const char * , const subst_list_type * );
void                   	 forward_model_free( forward_model_type * );
forward_model_type *   	 forward_model_alloc_copy(const forward_model_type * forward_model);
const char         *     forward_model_get_lsf_request(const forward_model_type * );
void                     forward_model_iset_job_arg( forward_model_type * forward_model , int job_index , const char * arg , const char * value);
ext_job_type           * forward_model_iget_job( forward_model_type * forward_model , int index);
void                     forward_model_set_lsf_request( forward_model_type * forward_model , const char* lsf_request );


#ifdef __cplusplus
}
#endif
#endif
