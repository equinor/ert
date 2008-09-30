#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <util.h>
#include <hash.h>
#include <ext_job.h>
#include <config.h>
#include <stringlist.h>

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
  char 	     * portable_exe;
  char 	     * target_file;
  char       * start_file;   /* Will not start if not this file is present */
  char 	     * stdout_file;
  char 	     * stdin_file;
  char 	     * stderr_file;
  stringlist_type * argv;  /* This should *NOT* start with the executable */
  stringlist_type * init_code;
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


const char * ext_job_get_name(const ext_job_type * ext_job) {
  return ext_job->name;
}




static ext_job_type * ext_job_alloc__(const char * name) {
  ext_job_type * ext_job = util_malloc(sizeof * ext_job , __func__);
  
  ext_job->__type_id    = __TYPE_ID__;
  ext_job->name         = util_alloc_string_copy(name);
  ext_job->portable_exe = NULL;
  ext_job->stdout_file  = NULL;
  ext_job->target_file  = NULL;
  ext_job->start_file   = NULL;
  ext_job->stdin_file   = NULL;
  ext_job->stderr_file  = NULL;
  ext_job->init_code    = NULL;
  ext_job->argv 	= NULL;
  ext_job->platform_exe = hash_alloc();
  ext_job->environment  = hash_alloc();

  return ext_job;
}


/* Exported function - must have name != NULL */
ext_job_type * ext_job_alloc(const char * name) {
  return ext_job_alloc__(name);
}


void ext_job_free(ext_job_type * ext_job) {
  free(ext_job->name);
  util_safe_free(ext_job->portable_exe);
  util_safe_free(ext_job->stdout_file);
  util_safe_free(ext_job->stdin_file);
  util_safe_free(ext_job->target_file);
  util_safe_free(ext_job->stderr_file);
  hash_free(ext_job->environment);
  hash_free(ext_job->platform_exe);

  if (ext_job->argv != NULL)      stringlist_free(ext_job->argv);
  if (ext_job->init_code != NULL) stringlist_free(ext_job->init_code);
  free(ext_job);
}

void ext_job_free__(void * __ext_job) {
  ext_job_free ( ext_job_safe_cast(__ext_job) );
}


void ext_job_set_portable_exe(ext_job_type * ext_job, const char * portable_exe) {
  ext_job->portable_exe = util_realloc_string_copy(ext_job->portable_exe , portable_exe);
}

void ext_job_set_stdout_file(ext_job_type * ext_job, const char * stdout_file) {
  ext_job->stdout_file = util_realloc_string_copy(ext_job->stdout_file , stdout_file);
}

void ext_job_set_target_file(ext_job_type * ext_job, const char * target_file) {
  ext_job->target_file = util_realloc_string_copy(ext_job->target_file , target_file);
}

void ext_job_set_start_file(ext_job_type * ext_job, const char * start_file) {
  ext_job->start_file = util_realloc_string_copy(ext_job->start_file , start_file);
}

void ext_job_set_name(ext_job_type * ext_job, const char * name) {
  ext_job->name = util_realloc_string_copy(ext_job->name , name);
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

void ext_job_add_environment(ext_job_type *ext_job , const char * key , const char * value) {
  hash_insert_hash_owned_ref( ext_job->environment , key , util_alloc_string_copy( value ) , free);
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

static void __fprintf_python_list(FILE * stream , const char * id , const stringlist_type * list , const hash_type * context_hash ) {
  int size;
  int i;
  fprintf(stream , "\"%s\" : " , id);
  fprintf(stream,"[");
  if (list == NULL)
    size = 0;
  else
    size = stringlist_get_size(list);

  for (i = 0; i < size; i++) {
    const char * value = stringlist_iget(list , i);
    
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
  char ** key_list = hash_alloc_keylist( (hash_type *) hash);
  fprintf(stream , "\"%s\" : " , id);
  fprintf(stream,"{");
  for (i = 0; i < size; i++) {
    const char * value = hash_get(hash , key_list[i]);

    fprintf(stream,"\"%s\" : " , key_list[i]);
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
  util_free_stringlist(key_list , size);
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
  __indent(stream, 0); __fprintf_python_string(stream , "name"  	  , ext_job->name , NULL);                    __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "portable_exe" 	  , ext_job->portable_exe , context_hash);    __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "target_file"  	  , ext_job->target_file , context_hash);     __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "start_file"  	  , ext_job->start_file , context_hash);      __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "stdout"    	  , ext_job->stdout_file , context_hash);     __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "stderr"    	  , ext_job->stderr_file , context_hash);     __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "stdin"     	  , ext_job->stdin_file , context_hash);      __end_line(stream);

  __indent(stream, 2); __fprintf_python_list(stream   , "argList"      	  , ext_job->argv      , context_hash);       __end_line(stream);
  __indent(stream, 2); __fprintf_python_list(stream   , "init_code"    	  , ext_job->init_code , context_hash);       __end_line(stream);
  __indent(stream, 2); __fprintf_python_hash(stream   , "environment"  	  , ext_job->environment , context_hash);     __end_line(stream);
  __indent(stream, 2); __fprintf_python_hash(stream   , "platform_exe" 	  , ext_job->platform_exe , context_hash); 
  fprintf(stream,"}");
}




static void ext_job_assert(const ext_job_type * ext_job) {
  bool OK = true;
  if (ext_job->name == NULL) {
    OK = false;
  }

  if (!OK) 
    util_abort("%s: errors in the ext_job instance. \n" , __func__);
}


ext_job_type * ext_job_fscanf_alloc(const char * name , const char * filename) {
  ext_job_type * ext_job = ext_job_alloc(name);
  config_type * config = config_alloc(  );
  
  {
    config_item_type * item;
    item = config_add_item(config , "STDIN"  	       , false , false); config_item_set_argc_minmax(item  , 1 , 1 , NULL);
    item = config_add_item(config , "STDOUT" 	       , false , false); config_item_set_argc_minmax(item  , 1 , 1 , NULL);
    item = config_add_item(config , "STDERR" 	       , false , false); config_item_set_argc_minmax(item  , 1 , 1 , NULL);
    item = config_add_item(config , "INIT_CODE"        , false , true ); config_item_set_argc_minmax(item  , 1 , 1 , NULL);
    item = config_add_item(config , "PORTABLE_EXE"     , false , false); config_item_set_argc_minmax(item  , 1 , 1 , NULL);
    item = config_add_item(config , "TARGET_FILE"      , false , false); config_item_set_argc_minmax(item  , 1 , 1 , NULL);
    item = config_add_item(config , "START_FILE"       , false , false); config_item_set_argc_minmax(item  , 1 , 1 , NULL);
    item = config_add_item(config , "ENV"              , false , true ); config_item_set_argc_minmax(item  , 2 , 2 , NULL);
    item = config_add_item(config , "PLATFORM_EXE"     , false , true ); config_item_set_argc_minmax(item  , 2 , 2 , NULL);
    item = config_add_item(config , "ARGLIST"          , false , true ); config_item_set_argc_minmax(item  , 1 ,-1 , NULL);
  }
  config_parse(config , filename , "--" , NULL , false , true);
  {
    if (config_item_set(config , "STDIN"))  	  ext_job_set_stdin_file(ext_job   , config_get(config  , "STDIN"));
    if (config_item_set(config , "STDOUT")) 	  ext_job_set_stdout_file(ext_job  , config_get(config  , "STDOUT"));
    if (config_item_set(config , "STDERR")) 	  ext_job_set_stderr_file(ext_job  , config_get(config  , "STDERR"));
    if (config_item_set(config , "TARGET_FILE"))  ext_job_set_target_file(ext_job  , config_get(config  , "TARGET_FILE"));
    if (config_item_set(config , "START_FILE"))   ext_job_set_start_file(ext_job   , config_get(config  , "START_FILE"));
    if (config_item_set(config , "PORTABLE_EXE")) ext_job_set_portable_exe(ext_job , config_get(config  , "PORTABLE_EXE"));

    if (config_item_set(config , "ARGLIST")) 
      ext_job->argv = config_alloc_complete_stringlist(config , "ARGLIST");
    
    if (config_item_set(config , "INIT_CODE")) 
      ext_job->init_code = config_alloc_complete_stringlist(config , "INIT_CODE");
    
    if (config_item_set(config , "ENV")) 
      ext_job->environment = config_alloc_hash(config , "ENV");
    
    if (config_item_set(config , "PLATFORM_EXE")) 
      ext_job->platform_exe = config_alloc_hash(config , "PLATFORM_EXE");
    
  }
  config_free(config);
  ext_job_assert(ext_job);
  return ext_job;
}




#undef ASSERT_TOKENS
#undef __TYPE_ID__
