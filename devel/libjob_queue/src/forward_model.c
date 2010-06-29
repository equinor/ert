#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <util.h>
#include <stdio.h>
#include <ext_job.h>
#include <ext_joblist.h>
#include <forward_model.h>
#include <subst_list.h>
#include <vector.h>
#include <parser.h>
/**
   This file implements a 'forward-model' object. I
*/



struct forward_model_struct {
  vector_type               * jobs;         /* The actual jobs in this forward model. */
  const ext_joblist_type    * ext_joblist;  /* This is the list of external jobs which have been installed - which we can choose from. */
  char                      * lsf_request;  /* The lsf_requests needed for this forward model == NULL if we are not using lsf. */
};

#define DEFAULT_JOB_MODULE   "jobs.py"
#define DEFAULT_JOBLIST_NAME "jobList"






forward_model_type * forward_model_alloc(const ext_joblist_type * ext_joblist, const char * lsf_request) {
  forward_model_type * forward_model = util_malloc( sizeof * forward_model , __func__);
  
  forward_model->jobs        = vector_alloc_new();
  forward_model->ext_joblist = ext_joblist;
  forward_model->lsf_request = NULL;
  
  if (lsf_request != NULL) 
    forward_model_set_lsf_request( forward_model , lsf_request );
  
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
    const ext_job_type * job = vector_iget_const( forward_model->jobs , i);
    stringlist_append_ref( names , ext_job_get_name( job ));
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
  ext_job_type * job = vector_iget( forward_model->jobs , job_index );
  ext_job_set_private_arg(job , arg , value);  
}


void forward_model_clear( forward_model_type * forward_model ) {
  vector_clear( forward_model->jobs );
}



void forward_model_free( forward_model_type * forward_model) {
  vector_free( forward_model->jobs );
  util_safe_free( forward_model->lsf_request );
  free(forward_model);
}




void forward_model_set_lsf_request( forward_model_type * forward_model , const char* lsf_request ) {
  forward_model->lsf_request = util_realloc_string_copy( forward_model->lsf_request , lsf_request );
}


/**
   this function takes an input string of the type:

   job1  job2  job3(arg1 = value1, arg2 = value2, arg3= value3)

   and creates a forward model of it. observe the following rules:
   
    * if the function takes private arguments it is not allowed with space
      between the end of the function name and the opening parenthesis.

*/


void forward_model_parse_init(forward_model_type * forward_model , const char * input_string ) {
  //tokenizer_type * tokenizer_alloc(" " , "\'\"" , ",=()" , null , null , null);
  //stringlist_type * tokens = tokenizer_buffer( tokenizer , input_string , true);
  //stringlist_free( tokens );
  //tokenizer_free( tokenizer );

  char * p1                          = (char *) input_string;
  while (true) {
    ext_job_type *  current_job;
    char         * job_name;
    int            job_index;          
    {
      int job_length  = strcspn(p1 , " (");  /* scanning until we meet ' ' or '(' */
      job_name = util_alloc_substring_copy(p1 , job_length);
      p1 += job_length;
    }
    job_index = vector_get_size( forward_model->jobs );
    current_job = forward_model_add_job(forward_model , job_name);

    if (*p1 == '(') {  /* the function has arguments. */
      int arg_length = strcspn(p1 , ")");
      if (arg_length == strlen(p1))
	util_abort("%s: paranthesis not terminated for job:%s \n",__func__ , job_name);
      {
	char  * arg_string          = util_alloc_substring_copy((p1 + 1) , arg_length - 1);
        ext_job_set_private_args_from_string( current_job , arg_string );
	p1 += (1 + arg_length);
      }
    } 
    /*****************************************************************/
    /* At this point we are done with the parsing - the rest of the
       code in this while { } construct is only to check that the
       input is well formed. */ 

    {
      int space_length = strspn(p1 , " ");
      p1 += space_length;
      if (*p1 == '(') 
	/* detected lonesome '(' */
	util_abort("%s: found space between job:%s and \'(\' - aborting \n",__func__ , job_name);
    }

    /* 
       now p1 should point at the next character after the job, 
       or after the ')' if the job has arguments.
    */
    
    if (*p1 == '\0') { /* we have parsed the whole string. */
      free(job_name);
      break;   
    }
  }
}



/*****************************************************************/

/*
  the name of the pyton module - and the variable in the module,
  used when running the remote jobs.
*/

void forward_model_python_fprintf(const forward_model_type * forward_model , const char * path, const subst_list_type * global_args) {
  char * module_file = util_alloc_filename(path , DEFAULT_JOB_MODULE , NULL);
  FILE * stream      = util_fopen(module_file , "w");
  int i;

  fprintf(stream , "%s = [" , DEFAULT_JOBLIST_NAME);
  for (i=0; i < vector_get_size(forward_model->jobs); i++) {
    const ext_job_type * job = vector_iget_const(forward_model->jobs , i);
    ext_job_python_fprintf(job , stream , global_args);
    if (i < (vector_get_size( forward_model->jobs ) - 1))
      fprintf(stream,",\n");
  }
  fprintf(stream , "]\n");
  fclose(stream);
  free(module_file);
}

#undef DEFAULT_JOB_MODULE   
#undef DEFAULT_JOBLIST_NAME 




forward_model_type * forward_model_alloc_copy(const forward_model_type * forward_model) {
  int ijob;
  forward_model_type * new;

  new = forward_model_alloc(forward_model->ext_joblist , forward_model->lsf_request );
  for (ijob = 0; ijob < vector_get_size(forward_model->jobs); ijob++) {
    const ext_job_type * job = vector_iget_const( forward_model->jobs , ijob);
    vector_append_owned_ref( new->jobs , ext_job_alloc_copy( job ) , ext_job_free__);
  }
  
  return new;
}

ext_job_type * forward_model_iget_job( forward_model_type * forward_model , int index) {
  return vector_iget( forward_model->jobs , index );
}



void forward_model_fprintf(const forward_model_type * forward_model , FILE * stream) {
  int ijob;
  for (ijob = 0; ijob < vector_get_size(forward_model->jobs); ijob++) {
    ext_job_fprintf( vector_iget(forward_model->jobs , ijob) , stream);
    fprintf(stream , "  ");
  }
  fprintf(stream , "\n");
}


const ext_joblist_type * forward_model_get_joblist(const forward_model_type * forward_model) {
  return forward_model->ext_joblist;
}


const char * forward_model_get_lsf_request(const forward_model_type * forward_model) {
  return forward_model->lsf_request;
}
