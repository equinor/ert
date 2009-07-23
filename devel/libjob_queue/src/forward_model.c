#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <util.h>
#include <stdio.h>
#include <ext_job.h>
#include <ext_joblist.h>
#include <forward_model.h>
#include <subst.h>
#include <lsf_request.h>
#include <vector.h>
#include <tokenizer.h>
/**
   This file implements a 'forward-model' object. I
*/



struct forward_model_struct {
  vector_type               * jobs;         /* The actual jobs in this forward model. */
  const ext_joblist_type    * ext_joblist;  /* This is the list of external jobs which have been installed - which we can choose from. */
  lsf_request_type          * lsf_request;  /* The lsf_requests needed for this forward model == NULL if we are not using lsf. */
};

#define DEFAULT_JOB_MODULE   "jobs.py"
#define DEFAULT_JOBLIST_NAME "jobList"
#define DEFAULT_START_TAG    "<"
#define DEFAULT_END_TAG      ">"



static char * __alloc_tagged_string(const char * s) {
  return util_alloc_sprintf("%s%s%s" , DEFAULT_START_TAG , s , DEFAULT_END_TAG);
}


/*****************************************************************/


static forward_model_type * forward_model_alloc__(const ext_joblist_type * ext_joblist, bool statoil_mode , bool use_lsf) {
  forward_model_type * forward_model = util_malloc( sizeof * forward_model , __func__);

  forward_model->jobs        = vector_alloc_new();
  forward_model->ext_joblist = ext_joblist;
  if (use_lsf)
    forward_model->lsf_request = lsf_request_alloc(statoil_mode);
  else
    forward_model->lsf_request = NULL;
  
  return forward_model;
}


/**
   This function adds the job named 'job_name' to the forward model. The return
   value is the newly created ext_job instance. This can bes used to set private
   arguments for this job.
*/

ext_job_type * forward_model_add_job(forward_model_type * forward_model , const char * job_name) {
  ext_job_type * new_job = ext_joblist_get_job_copy(forward_model->ext_joblist , job_name);
  vector_append_owned_ref( forward_model->jobs , new_job , ext_job_free__);
  return new_job;
}


void forward_model_free( forward_model_type * forward_model) {
  vector_free( forward_model->jobs );
  if (forward_model->lsf_request != NULL) 
    lsf_request_free(forward_model->lsf_request);
  free(forward_model);
}




static void forward_model_update_lsf_request(forward_model_type * forward_model) {
  if (forward_model->lsf_request != NULL) {
    int ijob;
    lsf_request_reset( forward_model->lsf_request );
    for (ijob = 0; ijob < vector_get_size(forward_model->jobs); ijob++) {
      bool last_job = false;
      const ext_job_type * job = vector_iget_const( forward_model->jobs , ijob);
      if (ijob == (vector_get_size(forward_model->jobs) - 1))
	last_job = true;
      lsf_request_update(forward_model->lsf_request , job , last_job);
    }
  }
}



/**
   This function takes an input string of the type:

   JOB1  JOB2  JOB3(arg1 = value1, arg2 = value2, arg3= value3)

   And creates a forward model of it. Observe the following rules:
   
    * If the function takes private arguments it is NOT allowed with space
      between the end of the function name and the opening parenthesis.

   
*/

forward_model_type * forward_model_alloc(const char * input_string , const ext_joblist_type * ext_joblist, bool statoil_mode , bool use_lsf) {
  forward_model_type * forward_model = forward_model_alloc__(ext_joblist , statoil_mode , use_lsf);
  //tokenizer_type * tokenizer_alloc(" " , "\'\"" , ",=()" , NULL , NULL);
  //stringlist_type * tokens = tokenizer_buffer( tokenizer , input_string , true);
  //stringlist_free( tokens );
  //tokenizer_free( tokenizer );

  char * p1                          = (char *) input_string;
  while (true) {
    char         * job_name;
    ext_job_type * job;
    {
      int job_length  = strcspn(p1 , " (");  /* Scanning until we meet ' ' or '(' */
      job_name = util_alloc_substring_copy(p1 , job_length);
      p1 += job_length;
    }
    job = forward_model_add_job(forward_model , job_name);
    
    if (*p1 == '(') {  /* The function has arguments. */
      int arg_length = strcspn(p1 , ")");
      if (arg_length == strlen(p1))
	util_abort("%s: paranthesis not terminated for job:%s \n",__func__ , job_name);
      {
	char  * arg_string = util_alloc_substring_copy((p1 + 1) , arg_length - 1);
	char ** key_value_list;
	int     num_arg, iarg;
	
	util_split_string(arg_string , "," , &num_arg , &key_value_list);
	for (iarg = 0; iarg < num_arg; iarg++) {
	  if (strchr(key_value_list[iarg] , '=') == NULL)
	    util_abort("%s: could not find \'=\' in argument string:%s \n",__func__ , key_value_list[iarg]);
	  
	  {
	    char * key , * value , *tagged_key;
	    char * tmp     = key_value_list[iarg];
	    int arg_length , value_length;
	    while (isspace(*tmp))  /* Skipping initial space */
	      tmp++;
	    
	    arg_length = strcspn(tmp , " =");
	    key  = util_alloc_substring_copy(tmp , arg_length);
	    tmp += arg_length;
	    while ((*tmp == ' ') || (*tmp == '='))
	      tmp++;
	    tagged_key = __alloc_tagged_string(key);

	    value_length = strcspn(tmp , " ");
	    value = util_alloc_substring_copy( tmp , value_length);

	    /* Setting the argument */
	    ext_job_set_private_arg(job , tagged_key , value);
	    free(key);
	    free(value);
	    free(tagged_key);
	    tmp += value_length;


	    /* Accept only trailing space - any other character indicates a failed parsing. */
	    while (*tmp != '\0') {
	      if (!isspace(*tmp))
		util_abort("%s: something wrong with:%s  - spaces are not allowed in key or value part.\n",__func__ , key_value_list[iarg]);
	      tmp++;
	    }
	  }
	}
	util_free_stringlist(key_value_list , num_arg);
	free(arg_string);
	p1 += (1 + arg_length);
      }
    } 
    
    {
      int space_length = strspn(p1 , " ");
      p1 += space_length;
      if (*p1 == '(') 
	/* Detected lonesome '(' */
	util_abort("%s: found space between job:%s and \'(\' - aborting \n",__func__ , job_name);
    }

    /* 
       Now p1 should point at the next character after the job, 
       or after the ')' if the job has arguments.
    */
    
    if (*p1 == '\0') { /* We have parsed the whole string. */
      free(job_name);
      break;   
    }
  }
  forward_model_update_lsf_request(forward_model);
  return forward_model;
}



/*****************************************************************/

/*
  The name of the pyton module - and the variable in the module,
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




forward_model_type * forward_model_alloc_copy(const forward_model_type * forward_model , bool statoil_mode) {
  int ijob;
  bool use_lsf = false;
  forward_model_type * new;

  if (forward_model->lsf_request != NULL)
    use_lsf = true;
  new = forward_model_alloc__(forward_model->ext_joblist , statoil_mode , use_lsf);
  for (ijob = 0; ijob < vector_get_size(forward_model->jobs); ijob++) {
    const ext_job_type * job = vector_iget_const( forward_model->jobs , ijob);
    vector_append_owned_ref( new->jobs , ext_job_alloc_copy( job ) , ext_job_free__);
  }
  forward_model_update_lsf_request(new);
  
  return new;
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
  if (forward_model->lsf_request != NULL)
    return lsf_request_get(forward_model->lsf_request);
  else
    return NULL;
}
