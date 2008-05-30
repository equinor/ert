#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <util.h>
#include <hash.h>
#include <ext_job.h>

/*

jobList = [
    {"portable_exe" : None, 
     "platform_exe" : {"x86_64" : "/local/eclipse/Geoquest/2006.2/bin/linux_x86_64/eclipse.exe",
                      "ia64"   : "/local/eclipse/Geoquest/2006.2/bin/linux_ia64/eclipse.exe"},
     "environment" : {"LM_LICENSE_FILE" : "1700@osl001lic.hda.hydro.com:1700@osl002lic.hda.hydro.com:1700@osl003lic.hda.hydro.com",
                      "F_UFMTENDIAN"    : "big"},
     "init_code" : "os.symlink(\"/local/eclipse/macros/ECL.CFG\" , \"ECL.CFG\")",
     "target_file":"222",
     "argList"   : [],
     "stdout"    : "eclipse.stdout",
     "stderr"    : "eclipse.stdout",
     "stdin"     : "eclipse.stdin"}]
*/


#define __TYPE_ID__ 763012


struct ext_job_struct {
  int          __type_id;
  char       * name;
  int  	       priority;
  char 	     * portable_exe;
  char 	     * init_code;
  char 	     * target_file;
  char 	     * stdout_file;
  char 	     * stdin_file;
  char 	     * stderr_file;
  char 	     ** argv; /* This should *NOT* start with the executable */
  int  	       argc;
  hash_type  * platform_exe;
  hash_type  * environment;
};


ext_job_type * ext_job_safe_cast(const void * __ext_job) {
  ext_job_type * ext_job = (ext_job_type * ) __ext_job;
  if (ext_job->__type_id != __TYPE_ID__) {
    util_abort("%s: safe_cast() failed - internal error\n",__func__);
    return NULL ;
  }  else
    return ext_job;
}


ext_job_type * ext_job_alloc(const char * name , int priority) {
  ext_job_type * ext_job = util_malloc(sizeof * ext_job , __func__);

  ext_job->__type_id = __TYPE_ID__;
  ext_job->name = util_alloc_string_copy(name);
  ext_job->priority     = priority;
  ext_job->portable_exe = NULL;
  ext_job->init_code = NULL;
  ext_job->stdout_file = NULL;
  ext_job->target_file = NULL;
  ext_job->stdin_file = NULL;
  ext_job->stderr_file = NULL;
  ext_job->argv = NULL;
  ext_job->argc = 0;
  ext_job->platform_exe = hash_alloc();
  ext_job->environment  = hash_alloc();

  return ext_job;
}


void ext_job_free(ext_job_type * ext_job) {
  free(ext_job->name);
  util_safe_free(ext_job->portable_exe);
  util_safe_free(ext_job->init_code);
  util_safe_free(ext_job->stdout_file);
  util_safe_free(ext_job->stdin_file);
  util_safe_free(ext_job->target_file);
  util_safe_free(ext_job->stderr_file);
  hash_free(ext_job->environment);
  hash_free(ext_job->platform_exe);
  
  util_free_string_list(ext_job->argv , ext_job->argc);
}



void ext_job_set_portable_exe(ext_job_type * ext_job, const char * portable_exe) {
  ext_job->portable_exe = util_realloc_string_copy(ext_job->portable_exe , portable_exe);
}

void ext_job_set_init_code(ext_job_type * ext_job, const char * init_code) {
  ext_job->init_code = util_realloc_string_copy(ext_job->init_code , init_code);
}

void ext_job_set_stdout_file(ext_job_type * ext_job, const char * stdout_file) {
  ext_job->stdout_file = util_realloc_string_copy(ext_job->stdout_file , stdout_file);
}

void ext_job_set_target_file(ext_job_type * ext_job, const char * target_file) {
  ext_job->target_file = util_realloc_string_copy(ext_job->target_file , target_file);
}


void ext_job_set_stdin_file(ext_job_type * ext_job, const char * stdin_file) {
  ext_job->stdin_file = util_realloc_string_copy(ext_job->stdin_file , stdin_file);
}

void ext_job_set_stderr_file(ext_job_type * ext_job, const char * stderr_file) {
  ext_job->stderr_file = util_realloc_string_copy(ext_job->stderr_file , stderr_file);
}

void ext_job_add_platform_exe(ext_job_type *ext_job , const char * platform , const char * exe) {
  hash_insert_hash_owned_ref( ext_job->platform_exe , platform , util_alloc_string_copy( exe ) , free);
}

void ext_job_add_environment(ext_job_type *ext_job , const char * platform , const char * exe) {
  hash_insert_hash_owned_ref( ext_job->environment , platform , util_alloc_string_copy( exe ) , free);
}




void ext_job_add_arg(ext_job_type * ext_job , const char * arg) {
  ext_job->argv = util_realloc(ext_job->argv , (ext_job->argc + 1) * sizeof * ext_job->argv , __func__);
  ext_job->argv[ext_job->argc] = util_alloc_string_copy( arg );
  ext_job->argc++;
}


static void __fprintf_python_string(FILE * stream , const char * id , const char * value, const hash_type * context_hash) {
  fprintf(stream , "\"%s\" : " , id);
  if (value == NULL)
    fprintf(stream,"None");
  else {
    
    if (context_hash != NULL) {
      fprintf(stream,"\"");
      util_filtered_fprintf( value , strlen(value) , stream , '<' , '>' , context_hash , util_filter_warn0);
      fprintf(stream,"\"");
    } else
      fprintf(stream,"\"%s\"" , value);
  }
}

static void __fprintf_python_list(FILE * stream , const char * id , const char ** list , int size, const hash_type * context_hash ) {
  int i;
  fprintf(stream , "\"%s\" : " , id);
  fprintf(stream,"[");
  for (i = 0; i < size; i++) {
    const char * value = list[i];

    if (context_hash != NULL) {
      fprintf(stream,"\"");
      util_filtered_fprintf( value , strlen(value) , stream , '<' , '>' , context_hash , util_filter_warn0);
      fprintf(stream,"\"");
    } else
      fprintf(stream,"\"%s\"" , value);

    if (i < (size - 1))
      fprintf(stream,",");
  }
  fprintf(stream,"]");
}


static void __fprintf_python_hash(FILE * stream , const char * id , const hash_type * hash, const hash_type * context_hash) {
  int i;
  int size = hash_get_size(hash);
  char ** key_list = hash_alloc_keylist(hash);
  fprintf(stream , "\"%s\" : " , id);
  fprintf(stream,"{");
  for (i = 0; i < size; i++) {
    const char * value = hash_get(hash , key_list[i]);

    fprintf(stream,"\"%s\":\"" , key_list[i]);
    if (context_hash != NULL) {
      fprintf(stream,"\"");
      util_filtered_fprintf( value , strlen(value) , stream , '<' , '>' , context_hash , util_filter_warn0);
      fprintf(stream,"\"");
    } else
      fprintf(stream,"\"%s\"" , value);
    
    if (i < (size - 1))
      fprintf(stream,",");
  }
  fprintf(stream,"}");
  util_free_string_list(key_list , size);
}




static void __end_line(FILE * stream) {
  fprintf(stream,",\n");
}

static void __indent(FILE * stream, int indent) {
  int i;
  for (i = 0; i < indent; i++)
    fprintf(stream," ");
}


void ext_job_python_fprintf(const ext_job_type * ext_job, FILE * stream, const hash_type * context_hash) { 
  fprintf(stream," {");
  __indent(stream, 0); __fprintf_python_string(stream , "portable_exe" 	  , ext_job->portable_exe , context_hash);    __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "init_code"    	  , ext_job->init_code , context_hash);       __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "target_file"  	  , ext_job->target_file , context_hash);     __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "stdout"    	  , ext_job->stdout_file , context_hash);     __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "stderr"    	  , ext_job->stderr_file , context_hash);     __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "stdin"     	  , ext_job->stdin_file , context_hash);      __end_line(stream);
  __indent(stream, 2); __fprintf_python_list(stream   , "argList"      	  , (const char **) ext_job->argv , ext_job->argc , context_hash); __end_line(stream);
  __indent(stream, 2); __fprintf_python_hash(stream   , "environment"  	  , ext_job->environment , context_hash);      __end_line(stream);
  __indent(stream, 2); __fprintf_python_hash(stream   , "platform_exe" 	  , ext_job->platform_exe , context_hash); 
  fprintf(stream,"}\n");
}


int ext_job_compare__(const void * __ext_job1, const void * __ext_job2 ) {
  ext_job_type * job1 = ext_job_safe_cast(__ext_job1);
  ext_job_type * job2 = ext_job_safe_cast(__ext_job2);

  if (job1->priority < job2->priority)
    return -1;
  else if (job1->priority == job2->priority) {
    util_abort("%s: two jobs %s and %s  have identitcal priority \n",__func__ , job1->name , job2->name);
    return 0;
  } else /* if (job1->priority > job2->priority) */
    return 1;
}


#undef __TYPE_ID__
