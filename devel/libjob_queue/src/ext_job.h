#ifndef __EXT_JOB_H__
#define __EXT_JOB_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <hash.h>
#include <stdio.h>
#include <subst_list.h>
#include <stringlist.h>

typedef struct ext_job_struct ext_job_type;

ext_job_type          * ext_job_alloc_copy(const ext_job_type * );
ext_job_type          * ext_job_alloc(const char * );
const char            * ext_job_get_name(const ext_job_type * );
const char            * ext_job_get_lsf_resources(const ext_job_type * );
void 	                ext_job_free(ext_job_type * ) ;
void 	                ext_job_free__(void * );
void 	                ext_job_add_environment(ext_job_type *, const char * , const char * ) ;
void                    ext_job_save( const ext_job_type * ext_job );
void                    ext_job_fprintf(const ext_job_type * , FILE * stream );
void                    ext_job_set_private_arg(ext_job_type * , const char *  , const char * );
void 	                ext_job_set_stdout_file(ext_job_type * , const char * );
void                    ext_job_set_config_file(ext_job_type * ext_job, const char * config_file);
void 	                ext_job_set_stdin_file(ext_job_type * , const char * );
void 	                ext_job_set_target_file(ext_job_type * , const char * );
void 	                ext_job_set_start_file(ext_job_type * , const char * );
void 	                ext_job_set_stderr_file(ext_job_type * , const char * );
void 	                ext_job_add_platform_exe(ext_job_type *, const char * , const char * ) ;
void 	                ext_job_set_argc(ext_job_type *   , const char ** , int);
void 	                ext_job_python_fprintf(const ext_job_type * , FILE * , const subst_list_type *);
ext_job_type          * ext_job_fscanf_alloc(const char * , const char * , const char *);
const stringlist_type * ext_job_get_arglist( const ext_job_type * ext_job );
bool                    ext_job_is_shared( const ext_job_type * ext_job );
bool                    ext_job_is_private( const ext_job_type * ext_job );

#ifdef __cplusplus
}
#endif
#endif
