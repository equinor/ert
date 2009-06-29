#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <util.h>
#include <hash.h>
#include <ext_job.h>
#include <config.h>
#include <stringlist.h>
#include <subst.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>



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
  int        	    __type_id;
  char       	  * name;
  char 	     	  * portable_exe;
  char 	     	  * target_file;
  char       	  * start_file;          /* Will not start if not this file is present */
  char 	     	  * stdout_file;
  char 	     	  * stdin_file;
  char 	     	  * stderr_file;
  char       	  * lsf_resources;  
  subst_list_type * private_args;     /* A substitution list of input arguments which is performed before the external substitutions. */
  stringlist_type * argv;             /* This should *NOT* start with the executable */
  stringlist_type * init_code;
  hash_type  	  * platform_exe;     /* The hash tables can NOT be NULL. */
  hash_type  	  * environment;
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
  
  ext_job->__type_id      = __TYPE_ID__;
  ext_job->name           = util_alloc_string_copy(name);
  ext_job->portable_exe   = NULL;
  ext_job->stdout_file    = NULL;
  ext_job->target_file    = NULL;
  ext_job->start_file     = NULL;
  ext_job->stdin_file     = NULL;
  ext_job->stderr_file    = NULL;
  ext_job->init_code      = NULL;
  ext_job->argv 	  = NULL;
  ext_job->lsf_resources  = NULL;
  ext_job->platform_exe   = NULL;
  ext_job->environment    = NULL;
  ext_job->argv           = NULL;
  ext_job->init_code      = NULL;
  /* 
     ext_job->private_args is set explicitly in the ext_job_alloc() 
     and ext_job_alloc_copy() functions. 
  */
  return ext_job;
}



/* Exported function - must have name != NULL */
ext_job_type * ext_job_alloc(const char * name) {
  ext_job_type * ext_job = ext_job_alloc__(name);
  ext_job->private_args  = subst_list_alloc();
  return ext_job;
}



/**
   Difficult to make a general hash_alloc_copy() which handles all
   possible variations of ownership+++ 
   
   This is a specialized implementation where it is assumed that all
   values in the hash are actuall pointers to \0 terminated strings.
*/


static hash_type * ext_job_hash_copyc__(hash_type * h) {
  if (h != NULL) {
    hash_type      * new_hash = hash_alloc();
    hash_iter_type * iter     = hash_iter_alloc( h);
    const char * key = hash_iter_get_next_key(iter);
    
    while (key != NULL) {
      char * value = hash_get( h , key);
      hash_insert_hash_owned_ref( new_hash , key , util_alloc_string_copy(value) , free);
      key = hash_iter_get_next_key(iter);
    }
    hash_iter_free(iter); 
    return new_hash;
  } else return NULL;
}

ext_job_type * ext_job_alloc_copy(const ext_job_type * src_job) {
  ext_job_type * new_job  = ext_job_alloc__(src_job->name);

  new_job->portable_exe   = util_alloc_string_copy(src_job->portable_exe);
  new_job->target_file    = util_alloc_string_copy(src_job->target_file);
  new_job->start_file     = util_alloc_string_copy(src_job->start_file);
  new_job->stdout_file    = util_alloc_string_copy(src_job->stdout_file);
  new_job->stdin_file     = util_alloc_string_copy(src_job->stdin_file);
  new_job->stderr_file    = util_alloc_string_copy(src_job->stderr_file);
  new_job->lsf_resources  = util_alloc_string_copy(src_job->lsf_resources);  

  if (src_job->argv      != NULL) new_job->argv          = stringlist_alloc_deep_copy( src_job->argv );
  if (src_job->init_code != NULL) new_job->init_code     = stringlist_alloc_deep_copy( src_job->init_code );
  
  new_job->platform_exe  = ext_job_hash_copyc__( src_job->platform_exe );
  new_job->environment   = ext_job_hash_copyc__( src_job->environment );
  new_job->private_args  = subst_list_alloc_deep_copy( src_job->private_args );
  return new_job;
}



void ext_job_free(ext_job_type * ext_job) {
  free(ext_job->name);
  util_safe_free(ext_job->portable_exe);
  util_safe_free(ext_job->stdout_file);
  util_safe_free(ext_job->stdin_file);
  util_safe_free(ext_job->target_file);
  util_safe_free(ext_job->stderr_file);
  util_safe_free(ext_job->lsf_resources);

  if (ext_job->environment != NULL)  hash_free(ext_job->environment);
  if (ext_job->platform_exe != NULL) hash_free(ext_job->platform_exe);
  
  if (ext_job->argv != NULL)         stringlist_free(ext_job->argv);
  if (ext_job->init_code != NULL)    stringlist_free(ext_job->init_code);
  
  subst_list_free( ext_job->private_args );
  free(ext_job);
}

void ext_job_free__(void * __ext_job) {
  ext_job_free ( ext_job_safe_cast(__ext_job) );
}


void ext_job_set_portable_exe(ext_job_type * ext_job, const char * portable_exe) {
  /**

     The portable exe can be a <...> string, i.e. not ready yet. Then
     we just have to trust the user to provide something sane in the
     end. If on the other hand portable_exe points to an existing file
     we:

      1. Call util_alloc_realpth() to get the full absolute path.
      2. Require that it is a executable file.
  
  */
  if (util_file_exists( portable_exe )) {
    char * full_path = util_alloc_realpath( portable_exe );
    if (!util_is_executable( full_path ))
      util_exit("%s: The program: %s -> %s is not executable \n",__func__ , portable_exe , full_path);
    else
      ext_job->portable_exe = util_realloc_string_copy(ext_job->portable_exe , full_path);
    free(full_path);
  } else
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

void ext_job_set_lsf_request(ext_job_type * ext_job, const char * lsf_request) {
  ext_job->lsf_resources = util_realloc_string_copy(ext_job->lsf_resources , lsf_request);
}


void ext_job_set_private_arg(ext_job_type * ext_job, const char * key , const char * value) {
  subst_list_insert_copy( ext_job->private_args  , key , value);
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


static void __fprintf_string(FILE * stream , const char * s , const subst_list_type * private_args, const subst_list_type * global_args) {
  char * tmp = subst_list_alloc_filtered_string( private_args , s ); /* internal filtering first */
  
  if (global_args != NULL) {
    fprintf(stream,"\"");
    subst_list_filtered_fprintf( global_args , tmp , stream );
    fprintf(stream,"\"");
  } else
    fprintf(stream,"\"%s\"" , tmp);

  free( tmp );
}


 
static void __fprintf_python_string(FILE * stream , const char * id , const char * value, const subst_list_type * private_args, const subst_list_type * global_args) {
  fprintf(stream , "\"%s\" : " , id);
  if (value == NULL)
    fprintf(stream,"None");
  else 
    __fprintf_string(stream , value , private_args , global_args);
}


static void __fprintf_python_list(FILE * stream , const char * id , const stringlist_type * list , const subst_list_type * private_args, const subst_list_type * global_args ) {
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
    __fprintf_string(stream , value , private_args , global_args);
    if (i < (size - 1))
      fprintf(stream,",");
  }
  fprintf(stream,"]");
}



static void __fprintf_python_hash(FILE * stream , const char * id , hash_type * hash, const subst_list_type * private_args, const subst_list_type * global_args) {
  fprintf(stream , "\"%s\" : " , id);
  fprintf(stream,"{");
  if (hash != NULL) {
    int   hash_size = hash_get_size(hash);
    int   counter   = 0;
    hash_iter_type * iter = hash_iter_alloc(hash);
    const char * key = hash_iter_get_next_key(iter);
    while (key != NULL) {
      const char * value = hash_get(hash , key);

      fprintf(stream,"\"%s\" : " , key);
      __fprintf_string(stream , value , private_args , global_args);
      
      if (counter < (hash_size - 1))
	fprintf(stream,",");
      
      key = hash_iter_get_next_key(iter);
    }
  }
  fprintf(stream,"}");
}




static void __end_line(FILE * stream) {
  fprintf(stream,",\n");
}

static void __indent(FILE * stream, int indent) {
  int i;
  for (i = 0; i < indent; i++)
    fprintf(stream," ");
}


void ext_job_python_fprintf(const ext_job_type * ext_job, FILE * stream, const subst_list_type * global_args) {
  fprintf(stream," {");
  __indent(stream, 0); __fprintf_python_string(stream , "name"  	  , ext_job->name , ext_job->private_args , NULL);               __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "portable_exe" 	  , ext_job->portable_exe , ext_job->private_args, global_args);  __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "target_file"  	  , ext_job->target_file  , ext_job->private_args, global_args);  __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "start_file"  	  , ext_job->start_file   , ext_job->private_args, global_args);  __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "stdout"    	  , ext_job->stdout_file  , ext_job->private_args, global_args);  __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "stderr"    	  , ext_job->stderr_file  , ext_job->private_args, global_args);  __end_line(stream);
  __indent(stream, 2); __fprintf_python_string(stream , "stdin"     	  , ext_job->stdin_file   , ext_job->private_args, global_args);  __end_line(stream);
  __indent(stream, 2); __fprintf_python_list(stream   , "argList"      	  , ext_job->argv         , ext_job->private_args, global_args);  __end_line(stream);
  __indent(stream, 2); __fprintf_python_list(stream   , "init_code"    	  , ext_job->init_code    , ext_job->private_args, global_args);  __end_line(stream);
  __indent(stream, 2); __fprintf_python_hash(stream   , "environment"  	  , ext_job->environment  , ext_job->private_args, global_args);  __end_line(stream);
  __indent(stream, 2); __fprintf_python_hash(stream   , "platform_exe" 	  , ext_job->platform_exe , ext_job->private_args, global_args);
  fprintf(stream,"}");
}


void ext_job_fprintf(const ext_job_type * ext_job , FILE * stream) {
  fprintf(stream , "%s(", ext_job->name);
  subst_list_fprintf(ext_job->private_args , stream);
  fprintf(stream , ")  ");
}



static void ext_job_assert(const ext_job_type * ext_job) {
  bool OK = true;
  if (ext_job->name == NULL) {
    OK = false;
  }

  if (!OK) 
    util_abort("%s: errors in the ext_job instance. \n" , __func__);
}


const char * ext_job_get_lsf_resources(const ext_job_type * ext_job) {
  return ext_job->lsf_resources;
}


ext_job_type * ext_job_fscanf_alloc(const char * name , const char * filename) {
  if (getuid() == util_get_file_uid( filename )) {
    mode_t mode        = util_get_file_mode( filename );
    mode_t target_mode = S_IRUSR + S_IWUSR + S_IRGRP + S_IWGRP + S_IROTH;
    if (mode != target_mode) {
      printf("** Updating mode on :\'%s\' to %d \n",filename , target_mode);
      chmod( filename , target_mode );
    }
  }
  
  if (util_file_readable( filename )) {
    ext_job_type * ext_job = ext_job_alloc(name);
    config_type * config   = config_alloc(  );
  
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
      item = config_add_item(config , "LSF_RESOURCES"    , false , false); config_item_set_argc_minmax(item  , 1 ,-1 , NULL);
    }
    config_parse(config , filename , "--" , NULL , NULL , NULL , false , true);
    {
      if (config_item_set(config , "STDIN"))  	    ext_job_set_stdin_file(ext_job   , config_get(config  , "STDIN"));
      if (config_item_set(config , "STDOUT")) 	    ext_job_set_stdout_file(ext_job  , config_get(config  , "STDOUT"));
      if (config_item_set(config , "STDERR")) 	    ext_job_set_stderr_file(ext_job  , config_get(config  , "STDERR"));
      if (config_item_set(config , "TARGET_FILE"))    ext_job_set_target_file(ext_job  , config_get(config  , "TARGET_FILE"));
      if (config_item_set(config , "START_FILE"))     ext_job_set_start_file(ext_job   , config_get(config  , "START_FILE"));
      if (config_item_set(config , "PORTABLE_EXE"))   ext_job_set_portable_exe(ext_job , config_get(config  , "PORTABLE_EXE"));

      if (config_item_set(config , "LSF_RESOURCES")) {
        char * lsf_resources = stringlist_alloc_joined_string(config_get_stringlist_ref(config , "LSF_RESOURCES") , " ");
        ext_job_set_lsf_request(ext_job   , lsf_resources);
        free(lsf_resources);
      }

      if (config_item_set(config , "ARGLIST")) 
        ext_job->argv = config_alloc_complete_stringlist(config , "ARGLIST");
        
      if (config_item_set(config , "INIT_CODE")) 
        ext_job->init_code = config_alloc_complete_stringlist(config , "INIT_CODE");

      /**
         The code assumes that the hash tables are valid, can not be NULL:
      */
    
      if (config_item_set(config , "ENV")) 
        ext_job->environment = config_alloc_hash(config , "ENV");

      if (config_item_set(config , "PLATFORM_EXE")) 
        ext_job->platform_exe = config_alloc_hash(config , "PLATFORM_EXE");
    
    }
    config_free(config);
    ext_job_assert(ext_job);
    return ext_job;
  } else {
    fprintf(stderr,"** Warning: you do not have permission to read file:\'%s\' - job:%s not available. \n", filename , name);
    return NULL;
  }
}




#undef ASSERT_TOKENS
#undef __TYPE_ID__
