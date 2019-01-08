/*
   Copyright (C) 2011  Statoil ASA, Norway.

   The file 'forward_model.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <unistd.h>

#include <ert/util/util.hpp>
#include <ert/util/vector.hpp>
#include <ert/util/parser.hpp>
#include <ert/res_util/subst_list.hpp>

#include <ert/job_queue/ext_job.hpp>
#include <ert/job_queue/ext_joblist.hpp>
#include <ert/job_queue/forward_model.hpp>
#include <ert/job_queue/job_status.hpp>

#include <ert/util/ecl_version.hpp>


/**
   This file implements a 'forward-model' object. I
*/

struct forward_model_struct {
  vector_type               * jobs;         /* The actual jobs in this forward model. */
  const ext_joblist_type    * ext_joblist;  /* This is the list of external jobs which have been installed - which we can choose from. */
};

#define DEFAULT_JOB_JSON     "jobs.json"
#define DEFAULT_STATUS_JSON  "status.json"
#define DEFAULT_JOB_MODULE   "jobs.py"
#define DEFAULT_JOBLIST_NAME "jobList"




forward_model_type * forward_model_alloc(const ext_joblist_type * ext_joblist) {
  forward_model_type * forward_model = (forward_model_type*)util_malloc( sizeof * forward_model );

  forward_model->jobs        = vector_alloc_new();
  forward_model->ext_joblist = ext_joblist;

  return forward_model;
}


/**
   Allocates and returns a stringlist with all the names in the
   current forward_model.
*/
stringlist_type * forward_model_alloc_joblist( const forward_model_type * forward_model ) {
  stringlist_type * names = stringlist_alloc_new( );
  int i;
  for (i=0; i < vector_get_size( forward_model->jobs ); i++) {
    const ext_job_type * job = (const ext_job_type*)vector_iget_const( forward_model->jobs , i);
    stringlist_append_copy( names , ext_job_get_name( job ));
  }

  return names;
}


/**
   This function adds the job named 'job_name' to the forward model. The return
   value is the newly created ext_job instance. This can be used to set private
   arguments for this job.
*/

ext_job_type * forward_model_add_job(forward_model_type * forward_model , const char * job_name) {
  ext_job_type * new_job = ext_joblist_get_job_copy(forward_model->ext_joblist , job_name);
  vector_append_owned_ref( forward_model->jobs , new_job , ext_job_free__);
  return new_job;
}



/**
   This function is used to set private argument values to jobs in the
   forward model (i.e. the argument values passed in with KEY=VALUE
   pairs in the defining ().

   The use of 'index' to get the job is unfortunate , however one
   forward model can contain several instances of the same job, it is
   therefor not possible to use name based lookup.
*/

void forward_model_iset_job_arg( forward_model_type * forward_model , int job_index , const char * arg , const char * value) {
  ext_job_type * job = (ext_job_type*)vector_iget( forward_model->jobs , job_index );
  ext_job_set_private_arg(job , arg , value);
}


void forward_model_clear( forward_model_type * forward_model ) {
  vector_clear( forward_model->jobs );
}



void forward_model_free( forward_model_type * forward_model) {
  vector_free( forward_model->jobs );
  free(forward_model);
}

/*
  Used with SIMULATION_JOB keyword
*/

void forward_model_parse_job_args(forward_model_type * forward_model, const stringlist_type * list) {

  stringlist_type * args = stringlist_alloc_deep_copy(list);
  const char * job_name = stringlist_iget(args, 0);
  ext_job_type * current_job = forward_model_add_job(forward_model , job_name);
  ext_job_free_deprecated_argv(current_job);
  stringlist_idel(args, 0);
  ext_job_set_args(current_job, args);
}

/**
   DEPRECATED, used with the old FORWARD_MODEL keyword

   this function takes an input string of the type:

   job3(arg1 = value1, arg2 = value2, arg3= value3)

   and adds a job to the forward. observe the following rules:

    * if the function takes private arguments it is not allowed with space
      between the end of the function name and the opening parenthesis.

*/

void forward_model_parse_job_deprecated_args(forward_model_type * forward_model, const char * input_string) {
  char * p1 = (char *) input_string;
  char * job_name;
  {
    int job_length  = strcspn(p1 , " (");  /* scanning until we meet ' ' or '(' */
    job_name = util_alloc_substring_copy(p1 , 0 , job_length);
    p1 += job_length;
  }

  ext_job_type * current_job = forward_model_add_job(forward_model , job_name);

  if (*p1 == '(') {  /* the function has arguments. */
    int arg_length = strcspn(p1 , ")");
    if (arg_length == strlen(p1))
      util_abort("%s: paranthesis not terminated for job:%s \n",__func__ , job_name);
   {
      char  * arg_string          = (char *)util_alloc_substring_copy((p1 + 1) , 0 , arg_length - 1);
      ext_job_set_private_args_from_string( current_job , arg_string );
      free( arg_string );
    }
  }

  free(job_name);
}



static void forward_model_json_fprintf(const forward_model_type * forward_model,
                                       const char * run_id,
                                       const char * path,
                                       const char * data_root,
                                       const subst_list_type * global_args,
                                       mode_t umask,
                                       const env_varlist_type * varlist) {
  char * json_file = (char*)util_alloc_filename(path , DEFAULT_JOB_JSON, NULL);
  FILE * stream    = util_fopen(json_file, "w");
  int i;

  fprintf(stream, "{\n");

  fprintf(stream, "\"umask\" : \"%04o\",\n", umask);
  fprintf(stream, "\"DATA_ROOT\": \"%s\",\n", data_root);
  env_varlist_json_fprintf(varlist, stream); fprintf(stream, ",\n");
  fprintf(stream, "\"jobList\" : [");
  for (i=0; i < vector_get_size(forward_model->jobs); i++) {
    const ext_job_type * job = (const ext_job_type*)vector_iget_const(forward_model->jobs , i);
    ext_job_json_fprintf(job , stream , global_args);
    if (i < (vector_get_size( forward_model->jobs ) - 1))
      fprintf(stream,",\n");
  }
  fprintf(stream, "],\n");

  fprintf(stream, "\n\"ert_version\" : [%d, %d, \"%s\"],\n",
          ecl_version_get_major_version(),
          ecl_version_get_minor_version(),
          ecl_version_get_micro_version());

  fprintf(stream, "\"run_id\" : \"%s\",\n", run_id);
  fprintf(stream, "\"ert_pid\" : \"%ld\"\n", (long)getpid()); //Long is big enough to hold __pid_t
  fprintf(stream, "}\n");
  fclose(stream);
  free(json_file);

  char * status_file = (char*)util_alloc_filename(path , DEFAULT_STATUS_JSON, NULL);
  remove(status_file);
  free(status_file);

}

void forward_model_formatted_fprintf(const forward_model_type * forward_model ,
                                     const char * run_id,
                                     const char * path,
                                     const char * data_root,
                                     const subst_list_type * global_args,
                                     mode_t umask,
                                     const env_varlist_type * list) {
  forward_model_json_fprintf(   forward_model, run_id, path, data_root, global_args, umask, list);
}

#undef DEFAULT_JOB_JSON
#undef DEFAULT_JOB_MODULE
#undef DEFAULT_JOBLIST_NAME




forward_model_type * forward_model_alloc_copy(const forward_model_type * forward_model) {
  int ijob;
  forward_model_type * new_;

  new_ = forward_model_alloc(forward_model->ext_joblist );
  for (ijob = 0; ijob < vector_get_size(forward_model->jobs); ijob++) {
    const ext_job_type * job = (const ext_job_type*)vector_iget_const( forward_model->jobs , ijob);
    vector_append_owned_ref( new_->jobs , ext_job_alloc_copy( job ) , ext_job_free__);
  }

  return new_;
}

ext_job_type * forward_model_iget_job( forward_model_type * forward_model , int index) {
  return (ext_job_type*)vector_iget( forward_model->jobs , index );
}



void forward_model_fprintf(const forward_model_type * forward_model , FILE * stream) {
  int ijob;
  fprintf(stream , " ");
  for (ijob = 0; ijob < vector_get_size(forward_model->jobs); ijob++) {
    ext_job_fprintf( (const ext_job_type*)vector_iget(forward_model->jobs , ijob) , stream);
    fprintf(stream , "  ");
  }
  fprintf(stream , "\n");
}


const ext_joblist_type * forward_model_get_joblist(const forward_model_type * forward_model) {
  return forward_model->ext_joblist;
}

int forward_model_get_length( const forward_model_type * forward_model ) {
  return vector_get_size( forward_model->jobs );
}
