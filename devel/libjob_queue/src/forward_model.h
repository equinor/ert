#ifndef __FORWARD_MODEL_H__
#define __FORWARD_MODEL_H__

#include <subst.h>
#include <ext_joblist.h>
#include <stdbool.h>

typedef struct  forward_model_struct forward_model_type ;

const ext_joblist_type * forward_model_get_joblist(const forward_model_type * );
void                   	 forward_model_fprintf(const forward_model_type *  , FILE * );
forward_model_type *   	 forward_model_alloc(const char * , const ext_joblist_type * , bool, bool);
void 	               	 forward_model_python_fprintf(const forward_model_type *  , const char * , const subst_list_type * );
void                   	 forward_model_free( forward_model_type * );
//void                   	 forward_model_set_private_arg(forward_model_type *  , const char *  , const char * );
forward_model_type *   	 forward_model_alloc_copy(const forward_model_type * forward_model , bool statoil_mode);
const char         *     forward_model_get_lsf_request(const forward_model_type * );
#endif
