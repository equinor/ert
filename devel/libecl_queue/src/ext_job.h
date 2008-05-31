#ifndef __EXT_JOB_H__
#define __EXT_JOB_H__
#include <hash.h>

typedef struct ext_job_struct ext_job_type;


ext_job_type * ext_job_alloc(const char * , int) ;
void ext_job_free(ext_job_type * ) ;
void ext_job_free__(void * );
void ext_job_add_environment(ext_job_type *, const char * , const char * ) ;
int  ext_job_get_priority__(const void * );

void ext_job_set_portable_exe(ext_job_type * , const char * );
void ext_job_set_init_code(ext_job_type * , const char * );
void ext_job_set_stdout_file(ext_job_type * , const char * );
void ext_job_set_stdin_file(ext_job_type * , const char * );
void ext_job_set_target_file(ext_job_type * , const char * );
void ext_job_set_stderr_file(ext_job_type * , const char * );
void ext_job_add_platform_exe(ext_job_type *, const char * , const char * ) ;
void ext_job_add_arg(ext_job_type *  , const char * );
void ext_job_python_fprintf(const ext_job_type * , FILE * , const hash_type *);

#endif
