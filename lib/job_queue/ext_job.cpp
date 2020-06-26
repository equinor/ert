/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'ext_job.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <ctype.h>
#include <string.h>
#include <stdio.h>

#include <ert/util/util.hpp>
#include <ert/util/hash.hpp>
#include <ert/util/stringlist.hpp>
#include <ert/util/parser.hpp>
#include <ert/res_util/subst_list.hpp>
#include <ert/res_util/res_env.hpp>

#include <ert/config/config_parser.hpp>
#include <ert/config/config_content.hpp>
#include <ert/config/config_error.hpp>

#include <ert/job_queue/job_kw_definitions.hpp>
#include <ert/job_queue/ext_job.hpp>


/*
  About arguments
  ---------------
  How a job is run is defined in terms of the following variables:

   o stdout_file / stdin_file / stderr_file
   o arglist
   o ....

  These variables will then contain string values from when the job
  configuration is read in, for example this little job

    STDOUT   my_stdout
    STDERR   my_stderr
    ARGLIST  my_job_input   my_job_output

  stdout & stderr are redirected to the files 'my_stdout' and
  'my_stderr' respectively, and when invoked with an exec() call the
  job is given the argumentlist:

       my_job_input   my_job_output

  This implies that _every_time_ this job is invoked the argumentlist
  will be identical; that is clearly quite limiting! To solve this we
  have the possibility of performing string substitutions on the
  strings in the job defintion prior to executing the job, this is
  handled with the privat_args substitutions. The definition for a
  copy-file job:


    EXECUTABLE   /bin/cp
    ARGLIST      <SRC_FILE>  <TARGET_FILE>


  This can then be invoked several times, with different key=value
  arguments for the SRC_FILE and TARGET_FILE:


      COPY_FILE(SRC_FILE = file1 , TARGET_FILE = /tmp/file1)
      COPY_FILE(SRC_FILE = file2 , TARGET_FILE = /tmp/file2)

*/
/*
  More on STDOUT/STDERR
  ---------------------
  If STDOUT/STDERR is is not defined, output is directed to:
  JOB_NAME.stdout.x or JOB_NAME.stderr.x, respectively

  STDOUT null   directs output to screen
  STDERR null   directs error messages to screen
*/
/*


jobList = [
    {"executable"  : None,
     "environment" : {"LM_LICENSE_PATH" : "1700@osl001lic.hda.hydro.com:1700@osl002lic.hda.hydro.com:1700@osl003lic.hda.hydro.com",
                      "F_UFMTENDIAN"    : "big"},
     "target_file":"222",
     "argList"   : [],
     "stdout"    : "eclipse.stdout",
     "stderr"    : "eclipse.stdout",
     "stdin"     : "eclipse.stdin"}]
*/


#define EXT_JOB_TYPE_ID 763012


#define EXT_JOB_STDOUT        "stdout"
#define EXT_JOB_STDERR        "stderr"
#define EXT_JOB_NO_STD_FILE   "null"    //Setting STDOUT null or STDERR null in forward model directs output to screen






struct ext_job_struct {
  UTIL_TYPE_ID_DECLARATION;
  char            * name;
  char            * executable;
  char            * target_file;
  char            * error_file;            /* Job has failed if this is present. */
  char            * start_file;            /* Will not start if not this file is present */
  char            * stdout_file;
  char            * stdin_file;
  char            * stderr_file;
  char            * license_path;          /* If this is NULL - it will be unrestricted ... */
  char            * license_root_path;
  char            * config_file;
  int               max_running;           /* 0 means unlimited. */
  int               max_running_minutes;   /* The maximum number of minutes this job is allowed to run - 0: unlimited. */

  int               min_arg;
  int               max_arg;
  int_vector_type * arg_types;
  stringlist_type * argv;                  /* Currently not in use, but will replace deprected_argv */
  subst_list_type * private_args;          /* A substitution list of input arguments which is performed before the external substitutions -
                                              these are the arguments supplied as key=value pairs in the forward model call. */
  char            * private_args_string;
  char            * argv_string;
  stringlist_type * deprecated_argv;       /* This should *NOT* start with the executable */
  hash_type       * environment;
  hash_type       * default_mapping;
  hash_type       * exec_env;
  char            * help_text;

  bool              private_job;           /* Can the current user/delete this job? (private_job == true) means the user can edit it. */
  bool              __valid;               /* Temporary variable consulted during the bootstrap - when the ext_job is completely initialized this should NOT be consulted anymore. */
};


static UTIL_SAFE_CAST_FUNCTION( ext_job , EXT_JOB_TYPE_ID)







static ext_job_type * ext_job_alloc__(const char * name , const char * license_root_path , bool private_job) {
  ext_job_type * ext_job = (ext_job_type*)util_malloc(sizeof * ext_job );

  UTIL_TYPE_ID_INIT( ext_job , EXT_JOB_TYPE_ID);
  ext_job->name                = util_alloc_string_copy( name );
  ext_job->license_root_path   = util_alloc_string_copy( license_root_path );
  ext_job->executable          = NULL;
  ext_job->stdout_file         = NULL;
  ext_job->target_file         = NULL;
  ext_job->error_file          = NULL;
  ext_job->start_file          = NULL;
  ext_job->stdin_file          = NULL;
  ext_job->stderr_file         = NULL;
  ext_job->environment         = hash_alloc();
  ext_job->default_mapping     = hash_alloc();
  ext_job->exec_env            = hash_alloc();
  ext_job->argv                = stringlist_alloc_new();
  ext_job->deprecated_argv     = NULL;
  ext_job->argv_string         = NULL;
  ext_job->__valid             = true;
  ext_job->license_path        = NULL;
  ext_job->config_file         = NULL;
  ext_job->max_running         = 0;                  /* 0 means unlimited. */
  ext_job->max_running_minutes = 0;                  /* 0 means unlimited. */
  ext_job->min_arg             = -1;
  ext_job->max_arg             = -1;
  ext_job->arg_types           = int_vector_alloc( 0 , CONFIG_STRING );
  ext_job->private_job         = private_job;        /* If private_job == true the job is user editable. */
  ext_job->help_text           = NULL;
  ext_job->private_args_string = NULL;

  /*
     ext_job->private_args is set explicitly in the ext_job_alloc()
     and ext_job_alloc_copy() functions.
  */
  return ext_job;
}


void ext_job_free_deprecated_argv(ext_job_type * ext_job) {
  if (ext_job->deprecated_argv) {
    stringlist_free(ext_job->deprecated_argv);
    ext_job->deprecated_argv = NULL;
  }
}

const char * ext_job_get_help_text( const ext_job_type * job ) {
  if (job->help_text != NULL)
    return job->help_text;
  else
    return "No help text installed for this job.";
}


void ext_job_set_help_text( ext_job_type * job , const char * help_text) {
  job->help_text = util_realloc_string_copy( job->help_text , help_text  );
}

/*
   Exported function - must have name != NULL. Observe that the
   instance returned from this function is not really usable for
   anything.

   Should probably define a minium set of parameters which must be set
   before the job is in a valid initialized state.
*/

ext_job_type * ext_job_alloc(const char * name , const char * license_root_path , bool private_job) {
  ext_job_type * ext_job = ext_job_alloc__(name , license_root_path , private_job);
  ext_job->private_args  = subst_list_alloc( NULL );
  return ext_job;
}



ext_job_type * ext_job_alloc_copy(const ext_job_type * src_job) {
  ext_job_type * new_job  = ext_job_alloc__( src_job->name , src_job->license_root_path , true /* All copies are by default private jobs. */);

  new_job->config_file    = util_alloc_string_copy(src_job->config_file);
  new_job->executable     = util_alloc_string_copy(src_job->executable);
  new_job->target_file    = util_alloc_string_copy(src_job->target_file);
  new_job->error_file     = util_alloc_string_copy(src_job->error_file);
  new_job->start_file     = util_alloc_string_copy(src_job->start_file);
  new_job->stdout_file    = util_alloc_string_copy(src_job->stdout_file);
  new_job->stdin_file     = util_alloc_string_copy(src_job->stdin_file);
  new_job->stderr_file    = util_alloc_string_copy(src_job->stderr_file);
  new_job->license_path   = util_alloc_string_copy(src_job->license_path);

  ext_job_set_help_text( new_job , src_job->help_text );

  new_job->max_running_minutes   = src_job->max_running_minutes;
  new_job->max_running           = src_job->max_running;
  new_job->min_arg               = src_job->min_arg;
  new_job->max_arg               = src_job->max_arg;
  new_job->arg_types             = int_vector_alloc_copy(src_job->arg_types);

  new_job->private_args          = subst_list_alloc_deep_copy( src_job->private_args );

  /* Copying over all the keys in the environment hash table */
  {
    hash_iter_type * iter     = hash_iter_alloc( src_job->environment );
    const char * key = hash_iter_get_next_key(iter);
    while (key != NULL) {
      char * value = (char*)hash_get( src_job->environment , key);
      if (value)
        hash_insert_hash_owned_ref( new_job->environment , key , util_alloc_string_copy(value) , free);
      else
        hash_insert_ref(new_job->environment, key, NULL);
      key = hash_iter_get_next_key(iter);
    }
    hash_iter_free(iter);
  }

  {
    hash_iter_type * iter     = hash_iter_alloc( src_job->exec_env );
    const char * key = hash_iter_get_next_key(iter);
    while (key != NULL) {
      char * value = (char*)hash_get( src_job->exec_env , key);
      if (value)
        hash_insert_hash_owned_ref( new_job->exec_env , key , util_alloc_string_copy(value) , free);
      else
        hash_insert_ref(new_job->exec_env, key, NULL);
      key = hash_iter_get_next_key(iter);
    }
    hash_iter_free(iter);
  }


  /* The default mapping. */
  {
    hash_iter_type * iter     = hash_iter_alloc( src_job->default_mapping );
    const char * key = hash_iter_get_next_key(iter);
    while (key != NULL) {
      char * value = (char*)hash_get( src_job->default_mapping , key);
      hash_insert_hash_owned_ref( new_job->default_mapping , key , util_alloc_string_copy(value) , free);
      key = hash_iter_get_next_key(iter);
    }
    hash_iter_free(iter);
  }


  if (src_job->deprecated_argv) {
    new_job->deprecated_argv = stringlist_alloc_new();
    stringlist_deep_copy( new_job->deprecated_argv , src_job->deprecated_argv );
  }

  return new_job;
}


void ext_job_free(ext_job_type * ext_job) {
  free(ext_job->name);
  free(ext_job->executable);
  free(ext_job->stdout_file);
  free(ext_job->stdin_file);
  free(ext_job->target_file);
  free(ext_job->error_file);
  free(ext_job->stderr_file);
  free(ext_job->license_path);
  free(ext_job->license_root_path);
  free(ext_job->config_file);
  free(ext_job->argv_string);
  free(ext_job->help_text);
  free(ext_job->private_args_string);

  hash_free( ext_job->default_mapping);
  hash_free( ext_job->environment );
  hash_free( ext_job->exec_env);

  stringlist_free(ext_job->argv);
  if (ext_job->deprecated_argv)
    stringlist_free(ext_job->deprecated_argv);
  subst_list_free( ext_job->private_args );

  int_vector_free(ext_job->arg_types);
  free(ext_job);
}

void ext_job_free__(void * __ext_job) {
  ext_job_free ( ext_job_safe_cast(__ext_job) );
}


static void __update_mode( const char * filename , mode_t add_mode) {
  util_addmode_if_owner( filename , add_mode);
}


/**
   The license_path =

   root_license_path / job_name / job_name

*/

static void ext_job_init_license_control(ext_job_type * ext_job) {
  if (ext_job->license_path == NULL) {
    ext_job->license_path   = util_alloc_sprintf("%s%c%s" , ext_job->license_root_path , UTIL_PATH_SEP_CHAR , ext_job->name );
    util_make_path( ext_job->license_path );
  }
}



void ext_job_set_max_time( ext_job_type * ext_job , int max_time ) {
  ext_job->max_running_minutes = max_time;
}



/**
  @executable parameter:
  The raw executable is either
    - an absolute path read directly from config
    - an absolute path constructed from the relative path from config
      with the assumption that the path was a relative path from the
      location of the job description file to the executable.

  @executable_raw parameter:
  The raw executable as read from config, unprocessed.

  This method have the following logic:

     @executable exists:
         We store the full path as the executable field of the job; and
         try to update the mode of the full_path executable to make sure it
         is executable.

     @executable does not exist, but @executable_raw exists:
        We have found an executable relative to the current working
        directory. This is deprecated behaviour, support will later be
        removed. Suggest new path to executable to user, relative to job
        description file and do a recursive call to this method, using
        the absolute path as @executable parameter

     @executable does not exist, @executable_raw does not exist and
     is an absolute path:
        Write error message

     @executable does not exist, @executable_raw does not exist and
     is a relative path:
        Search trough the PATH variable to try to locate the executable.
        If found, do a recursive call to this method, using the absolute path
        as @executable parameter

*/

void ext_job_set_executable(ext_job_type * ext_job, const char * executable_abs, const char * executable_input,bool search_path) {

  if (util_file_exists(executable_abs)) {
    /*
       The @executable parameter points to an existing file; we store
       the full path as the executable field of the job; we also try
       to update the mode of the full_path executable to make sure it
       is executable.
    */
    char * full_path = (char*)util_alloc_realpath( executable_abs );
    __update_mode( full_path , S_IRUSR + S_IWUSR + S_IXUSR + S_IRGRP + S_IWGRP + S_IXGRP + S_IROTH + S_IXOTH);  /* u:rwx  g:rwx  o:rx */
    ext_job->executable = util_realloc_string_copy(ext_job->executable , full_path);
    free( full_path );
  } else if (util_file_exists(executable_input)) {
    /*
       This "if" case means that we have found an executable relative
       to the current working directory. This is deprecated behaviour,
       support will be removed
    */
    char * full_path                  = (char*)util_alloc_abs_path(executable_input);
    const char * job_description_file = ext_job_get_config_file(ext_job);
    char * path_to_job_descr_file     = util_split_alloc_dirname(job_description_file);
    char * new_relative_path_to_exe   = (char*)util_alloc_rel_path(path_to_job_descr_file, full_path);
    char * relative_config_file       = (char*)util_alloc_rel_path(NULL , ext_job->config_file);

    fprintf(stderr,"/----------------------------------------------------------------\n");
    fprintf(stderr,"|                        ** WARNING **                            \n");
    fprintf(stderr,"|\n");
    fprintf(stderr,"| The convention for locating the executable in a forward model \n");
    fprintf(stderr,"| job has changed. When using a relative path in the EXECUTABLE \n");
    fprintf(stderr,"| setting in the job description file, the path will be interpreted\n");
    fprintf(stderr,"| relative to the location of the job description file. \n");
    fprintf(stderr,"|\n");
    fprintf(stderr,"| The job:\'%s\' will temporarilty continue to work in the \n",ext_job->name);
    fprintf(stderr,"| present form, but it is recommended to update: \n");
    fprintf(stderr,"|\n");
    fprintf(stderr,"|   1. Open the file:%s in an editor \n",relative_config_file);
    fprintf(stderr,"|\n");
    fprintf(stderr,"|   2. Change the EXECUTABLE line to: \n");
    fprintf(stderr,"|\n");
    fprintf(stderr,"|             EXECUTABLE  %s \n" , new_relative_path_to_exe);
    fprintf(stderr,"|\n");
    fprintf(stderr,"| The main advantage with this change in behaviour is that the\n");
    fprintf(stderr,"| job description file and the executable can be relocated.\n");
    fprintf(stderr,"\\----------------------------------------------------------------\n\n");

    ext_job_set_executable(ext_job, full_path, NULL, search_path);

    free(new_relative_path_to_exe);
    free(path_to_job_descr_file);
    free(full_path);
    free(relative_config_file);

   } else  if (util_is_abs_path( executable_input )) {
    /* If you have given an absolute path (i.e. starting with '/' to
       a non existing job we mark it as invalid - no possibility to
       provide context replacement afterwards. The job will be
       discarded by the calling scope.
    */
    fprintf(stderr , "** Warning: the executable:%s can not be found,\n"
                     "   job:%s will not be available.\n" , executable_abs , ext_job->name );
    ext_job->__valid = false;
  } else {
      if (search_path){
        /* Go through the PATH variable to try to locate the executable. */
        char * path_executable = res_env_alloc_PATH_executable( executable_input );

        if (path_executable != NULL) {
          ext_job_set_executable( ext_job , path_executable, NULL, search_path );
          free( path_executable );
        } else {
          /* We take the chance that user will supply a valid subst key for this later;
             if the final executable is not an actually executable file when exporting the
             job from ext_job_python_fprintf() a big warning will be written on stderr.
          */
          fprintf(stderr , "** Warning: Unable to locate the executable %s for job %s.\n"
                           "   Path to executable must be relative to the job description file, or an absolute path.\n"
                           "   Please update job EXECUTABLE for job %s. \n" , executable_abs , ext_job->name, ext_job->name);
          ext_job->__valid = false;
        }
      } else {
          ext_job->executable = util_realloc_string_copy(ext_job->executable , executable_input);
      }
  }

  /*
     If in the end we do not have execute rights to the executable :
     discard the job.
  */
  if (ext_job->executable != NULL) {
    if (util_file_exists(executable_abs)) {
      if (!util_is_executable( ext_job->executable )) {
        fprintf(stderr , "** You do not have execute rights to:%s - job will not be available.\n" , ext_job->executable);
        ext_job->__valid = false;  /* Mark the job as NOT successfully installed - the ext_job
                                      instance will later be freed and discarded. */
      }
    }
  }
}



/**
   Observe that this does NOT reread the ext_job instance from the new
   config_file.
*/


/*****************************************************************/
/* Scalar set and get functions                                  */

void ext_job_set_args(ext_job_type * ext_job, const stringlist_type * argv) {
  stringlist_deep_copy(ext_job->argv, argv);
}

void ext_job_set_config_file(ext_job_type * ext_job, const char * config_file) {
  ext_job->config_file = util_realloc_string_copy(ext_job->config_file , config_file);
}

const char * ext_job_get_config_file(const ext_job_type * ext_job) {
  return ext_job->config_file;
}

void ext_job_set_target_file(ext_job_type * ext_job, const char * target_file) {
  ext_job->target_file = util_realloc_string_copy(ext_job->target_file , target_file);
}

const char * ext_job_get_target_file(const ext_job_type * ext_job) {
  return ext_job->target_file;
}

void ext_job_set_error_file(ext_job_type * ext_job, const char * error_file) {
  ext_job->error_file = util_realloc_string_copy(ext_job->error_file , error_file);
}

const char * ext_job_get_error_file(const ext_job_type * ext_job) {
  return ext_job->error_file;
}

const char * ext_job_get_executable(const ext_job_type * ext_job) {
  return ext_job->executable;
}

void ext_job_set_start_file(ext_job_type * ext_job, const char * start_file) {
  ext_job->start_file = util_realloc_string_copy(ext_job->start_file , start_file);
}

const char * ext_job_get_start_file(const ext_job_type * ext_job) {
  return ext_job->start_file;
}

const char * ext_job_get_license_path(const ext_job_type * ext_job) {
  return ext_job->license_path;
}

const char * ext_job_get_name(const ext_job_type * ext_job) {
  return ext_job->name;
}
void ext_job_set_stdin_file(ext_job_type * ext_job, const char * stdin_file) {
  ext_job->stdin_file = util_realloc_string_copy(ext_job->stdin_file , stdin_file);
}

const char * ext_job_get_stdin_file(const ext_job_type * ext_job) {
  return ext_job->stdin_file;
}

void ext_job_set_stdout_file(ext_job_type * ext_job, const char * stdout_file) {
  if (!util_string_equal(stdout_file, EXT_JOB_NO_STD_FILE))
    ext_job->stdout_file = util_realloc_string_copy(ext_job->stdout_file , stdout_file);
}

const char * ext_job_get_stdout_file(const ext_job_type * ext_job) {
  return ext_job->stdout_file;
}

void ext_job_set_stderr_file(ext_job_type * ext_job, const char * stderr_file) {
  if (strcmp(stderr_file, EXT_JOB_NO_STD_FILE) != 0)
    ext_job->stderr_file = util_realloc_string_copy(ext_job->stderr_file , stderr_file);
}

const char * ext_job_get_stderr_file(const ext_job_type * ext_job) {
  return ext_job->stderr_file;
}

void ext_job_set_max_running( ext_job_type * ext_job , int max_running) {
  ext_job->max_running = max_running;
  if (max_running > 0)
    ext_job_init_license_control( ext_job );
}

int ext_job_get_max_running( const ext_job_type * ext_job ) {
  return ext_job->max_running;
}

void ext_job_set_max_running_minutes( ext_job_type * ext_job , int max_running_minutes) {
  ext_job->max_running_minutes = max_running_minutes;
}

int ext_job_get_max_running_minutes( const ext_job_type * ext_job ) {
  return ext_job->max_running_minutes;
}

static void ext_job_set_min_arg(ext_job_type * ext_job, int min_arg) {
  ext_job->min_arg = min_arg;
}

static void ext_job_set_max_arg(ext_job_type * ext_job, int max_arg) {
  ext_job->max_arg = max_arg;
}

int ext_job_get_min_arg(const ext_job_type * ext_job) {
  return ext_job->min_arg;
}

int ext_job_get_max_arg(const ext_job_type * ext_job) {
  return ext_job->max_arg;
}

/*****************************************************************/

void ext_job_set_private_arg(ext_job_type * ext_job, const char * key , const char * value) {
  subst_list_append_copy( ext_job->private_args  , key , value , NULL);
}

void ext_job_add_environment(ext_job_type *ext_job , const char * key , const char * value) {
  hash_insert_hash_owned_ref( ext_job->environment , key , util_alloc_string_copy( value ) , free);
}


void ext_job_clear_environment( ext_job_type * ext_job ) {
  hash_clear( ext_job->environment );
}

hash_type * ext_job_get_environment( ext_job_type * ext_job ) {
  return ext_job->environment;
}


/*****************************************************************/


static char * __alloc_filtered_string( const char * src_string , const subst_list_type * private_args, const subst_list_type * global_args) {
  char * tmp1 = subst_list_alloc_filtered_string( private_args , src_string ); /* internal filtering first */
  char * tmp2;

  if (global_args != NULL) {
    tmp2 = subst_list_alloc_filtered_string( global_args , tmp1 );        /* Global filtering. */
    free( tmp1 );
  } else
    tmp2 = tmp1;

  return tmp2;

}

static void __fprintf_string(FILE * stream , const char * s , const subst_list_type * private_args, const subst_list_type * global_args) {
  char * filtered_string = __alloc_filtered_string(s , private_args , global_args );
  fprintf(stream , "\"%s\"" , filtered_string );
  free( filtered_string );
}


static void __fprintf_python_string(FILE * stream ,
                                    const char * prefix,
                                    const char * id,
                                    const char * value,
                                    const char * suffix,
                                    const subst_list_type * private_args,
                                    const subst_list_type * global_args,
                                    const char * null_value) {
  fprintf(stream , "%s\"%s\" : " , prefix, id);
  if (value == NULL)
    fprintf(stream, "%s", null_value);
  else
    __fprintf_string(stream , value , private_args , global_args);
  fprintf(stream, "%s", suffix);
}


static void __fprintf_init_python_list( FILE * stream , const char * id ) {
  fprintf(stream , "\"%s\" : " , id);
  fprintf(stream,"[");
}

static void   __fprintf_close_python_list( FILE * stream ) {
  fprintf(stream,"]");
}



static hash_type * __alloc_filtered_hash(hash_type * input_hash,
                                         bool include_angular_values,
                                         const subst_list_type * private_args,
                                         const subst_list_type * global_args) {

  hash_type * output_hash = hash_alloc();
  hash_iter_type * iter = hash_iter_alloc(input_hash);
  const char * key = hash_iter_get_next_key(iter);
  while (key != NULL) {
    const char * value = (const char*) hash_get(input_hash , key);
    /*
      If the value is NULL or alternatively the special string value "null" we
      print the @null_value variable and continue.
    */
    if (!value || strcmp(value, "null") == 0)
      hash_insert_ref(output_hash, key, NULL);
    else {
      char * fv = __alloc_filtered_string(value, private_args, global_args);
      /*
        If the value string contains a <XXX> string which is not represented in
        the substitutionlists we will not print out the <xxxx> literal.
      */
      if (include_angular_values)
        hash_insert_hash_owned_ref(output_hash, key, fv, free);
      else {
        if ( !(fv[0] == '<' && fv[strlen(fv) -1] == '>') )
          hash_insert_hash_owned_ref(output_hash, key, fv, free);
      }

    }

    key = hash_iter_get_next_key(iter);
  }
  return output_hash;
}



static void __fprintf_python_hash(FILE * stream,
                                  const char * prefix,
                                  const char * id,
                                  hash_type * input_hash,
                                  const char * suffix,
                                  const subst_list_type * private_args,
                                  const subst_list_type * global_args,
                                  const char * null_value) {
  bool print_angular_values = false;
  hash_type * output_hash = __alloc_filtered_hash(input_hash, print_angular_values, private_args, global_args);
  int  hash_size = hash_get_size(output_hash);

  fprintf(stream , "%s\"%s\" : " , prefix, id);
  if (hash_size > 0) {
    bool first = true;
    fprintf(stream,"{");

    hash_iter_type * iter = hash_iter_alloc(output_hash);
    const char * key = hash_iter_get_next_key(iter);

    while (key != NULL) {
      const char * value = (const char*) hash_get(output_hash , key);
      if (!first)
        fprintf(stream,",");

      if (value)
        fprintf(stream,"\"%s\" : \"%s\"" , key, value);
      else
        fprintf(stream, "\"%s\" : %s", key, null_value);

      key = hash_iter_get_next_key(iter);
      first = false;
    }

    fprintf(stream,"}");
  } else
    fprintf(stream, "%s", null_value);
  fprintf(stream, "%s", suffix);

  hash_free(output_hash);
}


static void __fprintf_python_int(FILE * stream,
        const char * prefix,
        const char * key,
        int value,
        const char * suffix,
        const char * null_value) {
  fprintf(stream, "%s", prefix);
  if (value > 0)
    fprintf(stream , "\"%s\" : %d" , key , value);
  else
    fprintf(stream , "\"%s\" : %s" , key, null_value);
  fprintf(stream, "%s", suffix);
}

/*
   This is special cased to support the default mapping.
*/

static void __fprintf_python_argList(FILE * stream,
                                     const char * prefix,
                                     const ext_job_type * ext_job,
                                     const char * suffix,
                                     const subst_list_type * global_args) {

  stringlist_type * argv;
  if (ext_job->deprecated_argv)
    argv = ext_job->deprecated_argv;
  else
    argv = ext_job->argv;

  fprintf(stream, "%s", prefix);
  __fprintf_init_python_list( stream , "argList" );
  {
    for (int index = 0; index < stringlist_get_size( argv ); index++) {
      const char * src_string = stringlist_iget( argv , index );
      char * filtered_string = __alloc_filtered_string(src_string , ext_job->private_args , global_args );
      if (hash_has_key( ext_job->default_mapping , filtered_string ))
        filtered_string = (char *)util_realloc_string_copy( filtered_string , (const char *)hash_get( ext_job->default_mapping , filtered_string ));

      fprintf(stream , "\"%s\"" , filtered_string );
      if (index < (stringlist_get_size( argv) - 1))
        fprintf(stream , "," );

      free( filtered_string );
    }
  }
  __fprintf_close_python_list( stream );
  fprintf(stream, "%s", suffix);
}


static void __fprintf_python_arg_types(FILE * stream,
                                       const char * prefix,
                                       const char * key,
                                       const ext_job_type * ext_job,
                                       const char * suffix,
                                       const char * null_value) {
  fprintf(stream, "%s", prefix);
  if (!ext_job->arg_types) {
    fprintf(stream , "\"%s\" : %s" , key, null_value);
    goto postfix;
  }

  fprintf(stream, "\"%s\" : [", key);
  for (int i = 0; i < ext_job->max_arg; i++) {

    const char * arg_type = NULL;
    int type = int_vector_safe_iget(ext_job->arg_types, i);
    switch(type) {
    case CONFIG_INT:          arg_type = JOB_INT_TYPE; break;
    case CONFIG_FLOAT:        arg_type = JOB_FLOAT_TYPE; break;
    case CONFIG_STRING:       arg_type = JOB_STRING_TYPE; break;
    case CONFIG_BOOL:         arg_type = JOB_BOOL_TYPE; break;
    case CONFIG_RUNTIME_FILE: arg_type = JOB_RUNTIME_FILE_TYPE; break;
    case CONFIG_RUNTIME_INT:  arg_type = JOB_RUNTIME_INT_TYPE; break;
    default: util_abort("%s unknown config type %d", __func__, type);
    }

    fprintf(stream, "\"%s\"", arg_type);
    if ((i + 1) < ext_job->max_arg)
      fprintf(stream, ", ");
  }
  fprintf(stream, "]");

  postfix: fprintf(stream, "%s", suffix);
}


void ext_job_json_fprintf(const ext_job_type * ext_job, int job_index, FILE * stream, const subst_list_type * global_args) {
  const char * null_value = "null";
  
  char * file_stdout_index = NULL;
  char * file_stderr_index = NULL;
  
  file_stdout_index = util_alloc_sprintf("%s.%d",ext_job->stdout_file, job_index);
  file_stderr_index = util_alloc_sprintf("%s.%d",ext_job->stderr_file, job_index);
  
  fprintf(stream," {");
  {
    __fprintf_python_string(  stream, "",   "name",                ext_job->name,                ",\n", ext_job->private_args, NULL,        null_value);
    __fprintf_python_string(  stream, "  ", "executable",          ext_job->executable,          ",\n", ext_job->private_args, global_args, null_value);
    __fprintf_python_string(  stream, "  ", "target_file",         ext_job->target_file,         ",\n", ext_job->private_args, global_args, null_value);
    __fprintf_python_string(  stream, "  ", "error_file",          ext_job->error_file,          ",\n", ext_job->private_args, global_args, null_value);
    __fprintf_python_string(  stream, "  ", "start_file",          ext_job->start_file,          ",\n", ext_job->private_args, global_args, null_value);
    __fprintf_python_string(  stream, "  ", "stdout",              file_stdout_index,            ",\n", ext_job->private_args, global_args, null_value);
    __fprintf_python_string(  stream, "  ", "stderr",              file_stderr_index,            ",\n", ext_job->private_args, global_args, null_value);
    __fprintf_python_string(  stream, "  ", "stdin",               ext_job->stdin_file,          ",\n", ext_job->private_args, global_args, null_value);
    __fprintf_python_argList( stream, "  ",                        ext_job,                      ",\n",                        global_args            );
    __fprintf_python_hash(    stream, "  ", "environment",         ext_job->environment,         ",\n", ext_job->private_args, global_args, null_value);
    __fprintf_python_hash(    stream, "  ", "exec_env",            ext_job->exec_env,            ",\n", ext_job->private_args, global_args, null_value);
    __fprintf_python_string(  stream, "  ", "license_path",        ext_job->license_path,        ",\n", ext_job->private_args, global_args, null_value);
    __fprintf_python_int(     stream, "  ", "max_running_minutes", ext_job->max_running_minutes, ",\n",                                     null_value);
    __fprintf_python_int(     stream, "  ", "max_running",         ext_job->max_running,         ",\n",                                     null_value);
    __fprintf_python_int(     stream, "  ", "min_arg",             ext_job->min_arg,             ",\n",                                     null_value);

    __fprintf_python_arg_types( stream, "  ", "arg_types",           ext_job,                      ",\n",                                     null_value);

    __fprintf_python_int(     stream, "  ", "max_arg",             ext_job->max_arg,             "\n",                                      null_value);

  }
  fprintf(stream,"}");

  free( file_stdout_index );
  free( file_stderr_index );
}



#define PRINT_KEY_STRING( stream , key , value ) \
if (value != NULL)                               \
{                                                \
   fprintf(stream , "%16s ", key);               \
   fprintf(stream , "%s\n" , value);             \
}


#define PRINT_KEY_INT( stream , key , value ) \
if (value != 0)                               \
{                                             \
   fprintf(stream , "%16s ", key);            \
   fprintf(stream , "%d\n" , value);          \
}


/**
   Observe that the job will save itself to the internalized
   config_file; if you wish to save to some other place you must call
   ext_job_set_config_file() first.
*/

void ext_job_save( const ext_job_type * ext_job ) {
  FILE * stream = util_mkdir_fopen( ext_job->config_file , "w" );

  PRINT_KEY_STRING( stream , "EXECUTABLE"       , ext_job->executable);
  PRINT_KEY_STRING( stream , "STDIN"            , ext_job->stdin_file);
  PRINT_KEY_STRING( stream , "STDERR"           , ext_job->stderr_file);
  PRINT_KEY_STRING( stream , "STDOUT"           , ext_job->stdout_file);
  PRINT_KEY_STRING( stream , "TARGET_FILE"      , ext_job->target_file);
  PRINT_KEY_STRING( stream , "START_FILE"       , ext_job->start_file);
  PRINT_KEY_STRING( stream , "ERROR_FILE"       , ext_job->error_file);
  PRINT_KEY_INT( stream , "MAX_RUNNING"         , ext_job->max_running);
  PRINT_KEY_INT( stream , "MAX_RUNNING_MINUTES" , ext_job->max_running_minutes);

  stringlist_type * list;
  if (ext_job->deprecated_argv)
    list = ext_job->deprecated_argv;
  else
    list = ext_job->argv;

  if (stringlist_get_size( list) > 0) {
    fprintf(stream , "%16s" , "ARGLIST");
    stringlist_fprintf( list , " " , stream );
    fprintf(stream , "\n");
  }
  if (hash_get_size( ext_job->environment ) > 0) {
    hash_iter_type * hash_iter = hash_iter_alloc( ext_job->environment );
    while (!hash_iter_is_complete( hash_iter )) {
      const char * key = hash_iter_get_next_key( hash_iter );
      fprintf(stream, "%16s  %16s  %s\n" , "ENV" , key , (const char *) hash_get( ext_job->environment , key ));
    }
    hash_iter_free( hash_iter );
  }
  fclose( stream );
}

#undef PRINT_KEY_STRING
#undef PRINT_KEY_INT



void ext_job_fprintf(const ext_job_type * ext_job , FILE * stream) {
  fprintf(stream , "%s", ext_job->name);
  if (subst_list_get_size( ext_job->private_args ) > 0) {
    fprintf(stream , "(");
    subst_list_fprintf(ext_job->private_args , stream);
    fprintf(stream , ")");
  }
  fprintf(stream , "  ");
}


config_item_types ext_job_iget_argtype( const ext_job_type * ext_job, int index) {
  return (config_item_types)int_vector_safe_iget( ext_job->arg_types , index );
}


static void ext_job_iset_argtype_string( ext_job_type * ext_job , int iarg , const char * arg_type) {
  config_item_types type = job_kw_get_type(arg_type);
  if (type != CONFIG_INVALID)
    int_vector_iset( ext_job->arg_types , iarg , type );
}


ext_job_type * ext_job_fscanf_alloc(const char * name , const char * license_root_path , bool private_job , const char * config_file, bool search_path) {
  {
    mode_t target_mode = S_IRUSR + S_IWUSR + S_IRGRP + S_IWGRP + S_IROTH;  /* u+rw  g+rw  o+r */
    __update_mode( config_file , target_mode );
  }

  if (util_entry_readable( config_file)) {
    ext_job_type * ext_job = NULL;
    config_parser_type  * config  = config_alloc(  );

    {
      config_schema_item_type * item;
      item = config_add_schema_item(config , "MAX_RUNNING"         , false ); config_schema_item_set_argc_minmax(item  , 1 , 1 ); config_schema_item_iset_type( item , 0 , CONFIG_INT );
      item = config_add_schema_item(config , "STDIN"               , false ); config_schema_item_set_argc_minmax(item  , 1 , 1 );
      item = config_add_schema_item(config , "STDOUT"              , false ); config_schema_item_set_argc_minmax(item  , 1 , 1 );
      item = config_add_schema_item(config , "STDERR"              , false ); config_schema_item_set_argc_minmax(item  , 1 , 1 );
      item = config_add_schema_item(config , EXECUTABLE_KEY        , true ); config_schema_item_set_argc_minmax(item  , 1 , 1 ); config_schema_item_iset_type(item, 0, CONFIG_PATH);
      item = config_add_schema_item(config , "TARGET_FILE"         , false ); config_schema_item_set_argc_minmax(item  , 1 , 1 );
      item = config_add_schema_item(config , "ERROR_FILE"          , false ); config_schema_item_set_argc_minmax(item  , 1 , 1 );
      item = config_add_schema_item(config , "START_FILE"          , false ); config_schema_item_set_argc_minmax(item  , 1 , 1 );
      item = config_add_schema_item(config , "ENV"                 , false ); config_schema_item_set_argc_minmax(item  , 1 , 2 );
      item = config_add_schema_item(config , "EXEC_ENV"            , false ); config_schema_item_set_argc_minmax(item  , 1 , 2 );
      item = config_add_schema_item(config , "DEFAULT"             , false ); config_schema_item_set_argc_minmax(item  , 2 , 2 );
      item = config_add_schema_item(config , "ARGLIST"             , false ); config_schema_item_set_argc_minmax(item  , 1 , CONFIG_DEFAULT_ARG_MAX );
      item = config_add_schema_item(config , "MAX_RUNNING_MINUTES" , false ); config_schema_item_set_argc_minmax(item  , 1 , 1 ); config_schema_item_iset_type( item , 0 , CONFIG_INT );
      item = config_add_schema_item(config , MIN_ARG_KEY           , false ); config_schema_item_set_argc_minmax(item  , 1 , 1 ); config_schema_item_iset_type( item , 0 , CONFIG_INT );
      item = config_add_schema_item(config , MAX_ARG_KEY           , false ); config_schema_item_set_argc_minmax(item  , 1 , 1 ); config_schema_item_iset_type( item , 0 , CONFIG_INT );
      item = config_add_schema_item(config , ARG_TYPE_KEY          , false ); config_schema_item_set_argc_minmax( item , 2 , 2 ); config_schema_item_iset_type( item , 0 , CONFIG_INT );

      stringlist_type * var_types = stringlist_alloc_new();
      stringlist_append_copy(var_types, JOB_STRING_TYPE);
      stringlist_append_copy(var_types, JOB_INT_TYPE);
      stringlist_append_copy(var_types, JOB_FLOAT_TYPE);
      stringlist_append_copy(var_types, JOB_BOOL_TYPE);
      stringlist_append_copy(var_types, JOB_RUNTIME_FILE_TYPE);
      stringlist_append_copy(var_types, JOB_RUNTIME_INT_TYPE);

      config_schema_item_set_indexed_selection_set( item , 1 , var_types);
      stringlist_free(var_types);
    }
    config_add_alias(config , "EXECUTABLE" , "PORTABLE_EXE");
    {
      config_content_type * content = config_parse(config , config_file , "--" , NULL , NULL , NULL , CONFIG_UNRECOGNIZED_WARN , true);
      if (config_content_is_valid( content )) {
        ext_job = ext_job_alloc(name , license_root_path , private_job);
        ext_job_set_config_file( ext_job , config_file );


        if (config_content_has_item(content , "STDIN"))                 ext_job_set_stdin_file(ext_job       , config_content_iget(content  , "STDIN" , 0,0));
        if (config_content_has_item(content , "STDOUT"))                ext_job_set_stdout_file(ext_job , config_content_iget(content  , "STDOUT" , 0,0));
           else                                                         ext_job->stdout_file = util_alloc_filename( NULL, ext_job->name, EXT_JOB_STDOUT);

        if (config_content_has_item(content , "STDERR"))                ext_job_set_stderr_file(ext_job      , config_content_iget(content  , "STDERR" , 0,0));
           else                                                         ext_job->stderr_file = util_alloc_filename( NULL, ext_job->name, EXT_JOB_STDERR);

        if (config_content_has_item(content , "ERROR_FILE"))            ext_job_set_error_file(ext_job       , config_content_iget(content  , "ERROR_FILE" , 0,0));
        if (config_content_has_item(content , "TARGET_FILE"))           ext_job_set_target_file(ext_job      , config_content_iget(content  , "TARGET_FILE" , 0,0));
        if (config_content_has_item(content , "START_FILE"))            ext_job_set_start_file(ext_job       , config_content_iget(content  , "START_FILE" , 0,0));
        if (config_content_has_item(content , "MAX_RUNNING"))           ext_job_set_max_running(ext_job      , config_content_iget_as_int(content  , "MAX_RUNNING" , 0,0));
        if (config_content_has_item(content , "MAX_RUNNING_MINUTES"))   ext_job_set_max_time(ext_job         , config_content_iget_as_int(content  , "MAX_RUNNING_MINUTES" , 0,0));
        if (config_content_has_item(content , MIN_ARG_KEY))             ext_job_set_min_arg(ext_job          , config_content_iget_as_int(content  , MIN_ARG_KEY , 0,0));
        if (config_content_has_item(content , MAX_ARG_KEY))             ext_job_set_max_arg(ext_job          , config_content_iget_as_int(content  , MAX_ARG_KEY , 0,0));

        for (int i = 0; i < config_content_get_occurences( content , ARG_TYPE_KEY); i++) {
          int iarg = config_content_iget_as_int( content , ARG_TYPE_KEY , i , 0 );
          const char * arg_type = config_content_iget( content , ARG_TYPE_KEY , i , 1 );

          ext_job_iset_argtype_string( ext_job , iarg , arg_type );
        }

        {
          const char * executable     = config_content_get_value_as_executable(content  , EXECUTABLE_KEY);
          const char * executable_raw = config_content_iget(content  , EXECUTABLE_KEY , 0,0);
          ext_job_set_executable(ext_job , executable, executable_raw, search_path);
        }


        {
          if (config_content_has_item( content , "ARGLIST")) {
            ext_job->deprecated_argv = stringlist_alloc_new();
            config_content_node_type * arg_node = config_content_get_value_node( content , "ARGLIST");
            int i;
            for (i=0; i < config_content_node_get_size( arg_node ); i++)
              stringlist_append_copy( ext_job->deprecated_argv , config_content_node_iget( arg_node , i ));
          }
        }


        if (config_content_has_item( content , "ENV")) {
          const config_content_item_type * env_item = config_content_get_item( content , "ENV" );
          for (int ivar = 0; ivar < config_content_item_get_size( env_item ); ivar++) {
            const config_content_node_type * env_node = config_content_item_iget_node( env_item , ivar );
            const char * key   = config_content_node_iget( env_node, 0);
            if (config_content_node_get_size( env_node ) > 1) {
              const char * value = config_content_node_iget( env_node , 1);
              hash_insert_hash_owned_ref( ext_job->environment, key , util_alloc_string_copy( value ) , free);
            } else
              hash_insert_ref(ext_job->environment, key, NULL);
          }
        }

        if (config_content_has_item( content , "EXEC_ENV")) {
          const config_content_item_type * env_item = config_content_get_item( content , "EXEC_ENV" );
          for (int ivar = 0; ivar < config_content_item_get_size( env_item ); ivar++) {
            const config_content_node_type * env_node = config_content_item_iget_node( env_item , ivar );
            const char * key   = config_content_node_iget( env_node, 0);
            if (config_content_node_get_size( env_node ) > 1) {
              const char * value = config_content_node_iget( env_node , 1);
              hash_insert_hash_owned_ref( ext_job->exec_env, key , util_alloc_string_copy( value ) , free);
            } else
              hash_insert_ref(ext_job->exec_env, key, NULL);
          }
        }


        /* Default mappings; these are used to set values in the argList
           which have not been supplied by the calling context. */
        {
          if (config_content_has_item( content , "DEFAULT")) {
            const config_content_item_type * default_item = config_content_get_item( content , "DEFAULT");
            for (int ivar = 0; ivar < config_content_item_get_size( default_item ); ivar++) {
              const config_content_node_type * default_node = config_content_item_iget_node( default_item , ivar );
              for (int i=0; i < config_content_node_get_size( default_node ); i+= 2) {
                const char * key   = config_content_node_iget( default_node , i );
                const char * value = config_content_node_iget( default_node , i + 1);
                hash_insert_hash_owned_ref( ext_job->default_mapping, key , util_alloc_string_copy( value ) , free);
              }
            }
          }
        }

        if (!ext_job->__valid) {
          /*
            Something NOT OK (i.e. EXECUTABLE now); free the job instance and return NULL:
          */
          ext_job_free( ext_job );
          ext_job = NULL;
          fprintf(stderr,"** Warning: job: \'%s\' not available ... \n", name );
        }
      } else {
        config_error_type * error = config_content_get_errors( content );
        config_error_fprintf( error , true , stderr );
        fprintf(stderr,"** Warning: job: \'%s\' not available ... \n", name );
      }
      config_content_free( content );
    }
    config_free(config);

    return ext_job;
  } else {
    fprintf(stderr,"** Warning: you do not have permission to read file:\'%s\' - job:%s not available. \n", config_file , name);
    return NULL;
  }
}


const stringlist_type * ext_job_get_arglist( const ext_job_type * ext_job ) {
  if (ext_job->deprecated_argv)
    return ext_job->deprecated_argv;
  else
    return ext_job->argv;
}

/**
   Set the internal arguments of the job based on an input string
   @arg_string which is of the form:

       key1=value1, key2=value2 , key3=value3

   The internal private argument list is cleared before adding these
   arguments.
*/

int ext_job_set_private_args_from_string( ext_job_type * ext_job , const char * arg_string ) {
  subst_list_clear( ext_job->private_args );
  return subst_list_add_from_string( ext_job->private_args , arg_string , true );
}



bool ext_job_is_shared( const ext_job_type * ext_job ) {
  return !ext_job->private_job;
}

bool ext_job_is_private( const ext_job_type * ext_job ) {
  return ext_job->private_job;
}


#undef ASSERT_TOKENS
