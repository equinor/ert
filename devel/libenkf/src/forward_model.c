#include <stdlib.h>
#include <util.h>
#include <stdio.h>
#include <ext_job.h>
#include <ext_joblist.h>
#include <forward_model.h>

/**
   This file implements a 'forward-model' object. I
*/


/* Internal data type containing all the information needed to run on job. */

typedef struct forward_model_node_struct forward_model_node_type;
 
struct forward_model_node_struct {
  char  *  job_name;        /* The name of the job - this should be a key into the ext_joblist instance. */
}; 





struct forward_model_struct {
  int 	  alloc_size;        	    /* The number of elements in the jobs vector. */
  int 	  size;              	    /* The number of elements in this forward_model. */
  forward_model_node_type ** jobs;  /* The actual job nodes. */
};


static forward_model_node_type * forward_model_node_alloc( const char * job_name) {
  forward_model_node_type * node = util_malloc( sizeof * node , __func__);
  node->job_name = util_alloc_string_copy( job_name );
  return node;
}

							   
static void forward_model_node_free( forward_model_node_type * node) {
  free(node->job_name);
  free(node);
}




/*****************************************************************/

static void forward_model_realloc(forward_model_type * forward_model , int new_alloc_size) {
  forward_model->jobs = util_realloc(forward_model->jobs , new_alloc_size * sizeof * forward_model->jobs, __func__);
  {
    /* Initializing the new node pointers. */
    int i;
    for (i=forward_model->alloc_size; i < new_alloc_size; i++)
      forward_model->jobs[i] = NULL;
  }
  forward_model->alloc_size = new_alloc_size;
}



forward_model_type * forward_model_alloc() {
  forward_model_type * forward_model = util_malloc( sizeof * forward_model , __func__);

  forward_model->alloc_size = 0;
  forward_model->size       = 0;
  forward_model->jobs       = NULL; 
  
  return forward_model;
}


static void forward_model_add_node__(forward_model_type * forward_model , forward_model_node_type * node) {
  if (forward_model->size == forward_model->size)
    forward_model_realloc(forward_model , 2 * (forward_model->alloc_size + 1));
  forward_model->jobs[forward_model->size] = node;
  forward_model->size++;
}



void forward_model_free( forward_model_type * forward_model) {

  int ijob;
  for (ijob = 0; ijob < forward_model->size; ijob++)
    forward_model_node_free( forward_model->jobs[ijob] );
  util_safe_free(forward_model->jobs);
  free(forward_model);
}
