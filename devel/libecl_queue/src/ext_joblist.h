#ifndef __EXT_JOBLIST_H__
#define __EXT_JOBLIST_H__
#include <hash.h>
#include <ext_job.h>

typedef struct ext_joblist_struct ext_joblist_type;

ext_joblist_type * ext_joblist_alloc();
void               ext_joblist_free(ext_joblist_type * );
void               ext_joblist_add_job(ext_joblist_type * , const ext_job_type *);
ext_job_type 	 * ext_joblist_get_job(const ext_joblist_type * , const char * );
void               ext_joblist_python_fprintf(const ext_joblist_type * , const char ** , int , const char * , const hash_type * );
ext_job_type     * ext_joblist_alloc_new(ext_joblist_type * , const char * );
bool               ext_joblist_has_job(const ext_joblist_type *  , const char * );

#endif

