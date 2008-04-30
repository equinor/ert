#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <ecl_queue.h>
#include <util.h>
#include <ecl_util.h>
#include <basic_queue_driver.h>
#include <path_fmt.h>
#include <pthread.h>


/**

All queue drivers must support the following functions:

  submit: This will submit a job, and return a pointer to a 
          newly allocated queue_job instance.

  clean:  This will clear up all resources used by the job.

  abort:  This will stop the job, and then call clean.

  status: This will get the status of the job.
*/ 


typedef enum {submit_OK = 0 , submit_job_FAIL , submit_driver_FAIL} submit_status_type;

typedef struct {
  int                  	 external_id;
  int                  	 submit_attempt; 
  char                 	*target_file;
  char                 	*ecl_base;
  char                 	*run_path;
  char                  *smspec_file;
  ecl_job_status_type  	 job_status;
  basic_queue_job_type 	*job_data;
} ecl_queue_node_type;

/*****************************************************************/

static void ecl_queue_node_clear(ecl_queue_node_type * node) {
  node->external_id    = -1;
  node->job_status     = ecl_queue_null;
  node->submit_attempt = 0;
  node->target_file    = NULL;
  node->ecl_base       = NULL;
  node->run_path       = NULL;
  node->job_data       = NULL;
  node->smspec_file    = NULL;
}


static ecl_queue_node_type * ecl_queue_node_alloc() {
  ecl_queue_node_type * node = util_malloc(sizeof * node , __func__);
  ecl_queue_node_clear(node);
  return node;
}


static void ecl_queue_node_set_status(ecl_queue_node_type * node, ecl_job_status_type status) {
  node->job_status = status;
}

static void ecl_queue_node_free_data(ecl_queue_node_type * node) {
  if (node->ecl_base != NULL)    free(node->ecl_base);
  if (node->run_path != NULL)    free(node->run_path);
  if (node->target_file != NULL) free(node->target_file);
  if (node->smspec_file != NULL) free(node->smspec_file);
  if (node->job_data != NULL) 
    util_abort("%s: internal error - driver spesific job data has not been freed - will leak.\n",__func__);
}

static void ecl_queue_node_free(ecl_queue_node_type * node) {
  ecl_queue_node_free_data(node);
  free(node);
}

static ecl_job_status_type ecl_queue_node_get_status(const ecl_queue_node_type * node) {
  return node->job_status;
}


static int ecl_queue_node_get_external_id(const ecl_queue_node_type * node) {
  if (node->external_id < 0) 
    util_abort("%s: tried to get external id from uninitialized job - aborting \n",__func__);
  
  return node->external_id;
}


static void ecl_queue_node_finalize(ecl_queue_node_type * node) {
  ecl_queue_node_free_data(node);
  ecl_queue_node_clear(node);
}



/*****************************************************************/

struct ecl_queue_struct {
  int                        target_report;
  int                        sleep_time;
  int                        active_size; 
  int                        size;
  int                        max_submit;
  int                        max_running; 
  char                     * submit_cmd;
  char                     * eclipse_exe;
  char                     * eclipse_config;
  char                     * license_server;
  char                     * eclipse_LD_path;
  path_fmt_type            * run_path_fmt;
  path_fmt_type            * ecl_base_fmt;
  path_fmt_type            * target_file_fmt;
  ecl_queue_node_type     ** jobs;
  basic_queue_driver_type  * driver;
  pthread_mutex_t            active_mutex;
  pthread_mutex_t            status_mutex;
  int                        status_list[ecl_queue_max_state];
};

static bool ecl_queue_change_node_status(ecl_queue_type *  , ecl_queue_node_type *  , ecl_job_status_type );


static void ecl_queue_initialize_node(ecl_queue_type * queue , int queue_index , int external_id , int target_report) {
  if (external_id < 0) 
    util_abort("%s: external_id must be >= 0 - aborting \n",__func__);
  {
    ecl_queue_node_type * node = queue->jobs[queue_index];
    node->external_id    = external_id;
    node->submit_attempt = 0;
    node->run_path       = path_fmt_alloc_path(queue->run_path_fmt , external_id);

    node->ecl_base       = path_fmt_alloc_path(queue->ecl_base_fmt , external_id);
    if (queue->target_file_fmt != NULL)
      node->target_file = path_fmt_alloc_path(queue->target_file_fmt , external_id , external_id , target_report);
    else
      node->target_file = NULL;
    {
      bool fmt_file = false;
      node->smspec_file   = ecl_util_alloc_filename(node->run_path , node->ecl_base , ecl_summary_header_file , fmt_file , -1);
    }
    node->job_data      = NULL;
    if ( !util_path_exists(node->run_path) ) 
      util_abort("%s: the run_path: %s does not exist - aborting \n",__func__ , node->run_path);
    
    ecl_queue_change_node_status(queue , node , ecl_queue_waiting);
  }
}


static int ecl_queue_get_active_size(ecl_queue_type * queue) {
  int active_size;
  pthread_mutex_lock( &queue->active_mutex );
  active_size = queue->active_size;
  pthread_mutex_unlock( &queue->active_mutex );
  return active_size;
}


static void ecl_queue_assert_queue_index(const ecl_queue_type * queue , int queue_index) {
  if (queue_index < 0 || queue_index >= queue->size) 
    util_abort("%s: invalid queue_index - internal error - aborting \n",__func__);
}


static bool ecl_queue_change_node_status(ecl_queue_type * queue , ecl_queue_node_type * node , ecl_job_status_type new_status) {
  ecl_job_status_type old_status = ecl_queue_node_get_status(node);
  ecl_queue_node_set_status(node , new_status);
  queue->status_list[old_status]--;
  queue->status_list[new_status]++;
  if (new_status != old_status)
    return true;
  else
    return false;
}

static void ecl_queue_free_job(ecl_queue_type * queue , ecl_queue_node_type * node) {
  basic_queue_driver_type *driver  = queue->driver;
  driver->free_job(driver , node->job_data);
  node->job_data = NULL;
}


static void ecl_queue_update_status(ecl_queue_type * queue ) {
  basic_queue_driver_type *driver  = queue->driver;
  int ijob;
  for (ijob = 0; ijob < queue->size; ijob++) {
    ecl_queue_node_type * node       = queue->jobs[ijob];
    if (node->job_data != NULL) {
      ecl_job_status_type  new_status = driver->get_status(driver , node->job_data);
      ecl_queue_change_node_status(queue , node , new_status);
    }
  }
}


static submit_status_type ecl_queue_submit_job(ecl_queue_type * queue , int queue_index) {
  submit_status_type submit_status;
  ecl_queue_assert_queue_index(queue , queue_index);
  ecl_queue_node_type * node = queue->jobs[queue_index];
  basic_queue_driver_type *driver  = queue->driver;
  if (node->submit_attempt < queue->max_submit) {
    if (util_file_exists(node->smspec_file))
      util_unlink_existing(node->smspec_file);
    {
      basic_queue_job_type * job_data = driver->submit(queue->driver         , 
						       queue_index           , 
						       queue->submit_cmd     , 
						       node->run_path        , 
						       node->ecl_base        , 
						       queue->eclipse_exe    ,  
						       queue->eclipse_config , 
						       queue->eclipse_LD_path,
						       queue->license_server);
      if (job_data != NULL) {
	ecl_queue_change_node_status(queue , node , driver->get_status(driver , node->job_data));
	node->job_data = job_data;
	node->submit_attempt++;
	submit_status = submit_OK;
      } else
	submit_status = submit_driver_FAIL;
    }
  } else {
    ecl_queue_change_node_status(queue , node , ecl_queue_complete_FAIL);
    submit_status = submit_job_FAIL;
  }
  return submit_status;
}

static void ecl_queue_print_status(const ecl_queue_type * queue) {
  printf("Target report.....: %d \n",queue->target_report);
  printf("active_size ......: %d \n",queue->active_size);
}


ecl_job_status_type ecl_queue_export_job_status(ecl_queue_type * queue , int external_id) {
  bool node_found    = false;
  int active_size    = ecl_queue_get_active_size(queue);
  int queue_index    = 0; 
  ecl_job_status_type status;
  while (queue_index < active_size) {
    ecl_queue_node_type * node = queue->jobs[queue_index];
    if (ecl_queue_node_get_external_id(node) == external_id) {
      node_found = true;
      status = node->job_status;
      break;
    } else
      queue_index++;
  }
  
  if (node_found)
    return status;
  else {
    ecl_queue_print_status(queue);
    util_abort("%s: could not find job with id: %d - aborting.\n",__func__ , external_id);
    return status; /* Dummy --- */
  }
}


static void ecl_queue_print_jobs(const ecl_queue_type *queue) {
  int waiting  = queue->status_list[ecl_queue_waiting];
  int pending  = queue->status_list[ecl_queue_pending];
  /* 
     EXIT and DONE are included in "xxx_running", because the target
     file has not yet been checked.
  */
  int running  = queue->status_list[ecl_queue_running] + queue->status_list[ecl_queue_done] + queue->status_list[ecl_queue_exit];
  int complete = queue->status_list[ecl_queue_complete_OK];
  int failed   = queue->status_list[ecl_queue_complete_FAIL];
  int restarts = queue->status_list[ecl_queue_restart];  

  printf("Waiting: %3d    Pending: %3d    Running: %3d     Restarts: %3d    Failed: %3d   Complete: %3d   [ ]\b",waiting , pending , running , restarts , failed , complete);
  fflush(stdout);
}


void ecl_queue_finalize(ecl_queue_type * queue) {
  int i;
  for (i=0; i < queue->size; i++) 
    ecl_queue_node_finalize(queue->jobs[i]);
  
  for (i=0; i < ecl_queue_max_state; i++)
    queue->status_list[i] = 0;
  
  queue->active_size = 0;
  rsh_driver_summarize(queue->size);
}



void  ecl_queue_run_jobs(ecl_queue_type * queue , int num_total_run) {
  int phase = 0;
  int queue_index;
  int old_status_list[ecl_queue_max_state];
  {
    int i;
    for (i=0; i < ecl_queue_max_state; i++)
      old_status_list[i] = 0;
  }
  
  do {
    char spinner[4];
    spinner[0] = '-';
    spinner[1] = '\\';
    spinner[2] = '|';
    spinner[3] = '/';
    ecl_queue_update_status(queue);
    if (memcmp(old_status_list , queue->status_list , ecl_queue_max_state * sizeof * old_status_list) != 0) {
      printf("\b \n");
      ecl_queue_print_jobs(queue);
      memcpy(old_status_list , queue->status_list , ecl_queue_max_state * sizeof * old_status_list);
    } else {
      printf("\b%c",spinner[phase]); 
      fflush(stdout);
      phase = (phase + 1) % 4;
    }
    
    {
      /* Submitting new jobs */
      int active_size    = ecl_queue_get_active_size(queue);
      int total_active   = queue->status_list[ecl_queue_pending] + queue->status_list[ecl_queue_running];
      int num_submit_new = queue->max_running - total_active;
      queue_index = 0;
      while ((queue_index < active_size) && (num_submit_new > 0)) {
	ecl_queue_node_type * node = queue->jobs[queue_index];
	if (ecl_queue_node_get_status(node) == ecl_queue_waiting) {
	  submit_status_type submit_status = ecl_queue_submit_job(queue , queue_index);
	  if (submit_status == submit_OK) 
	    num_submit_new--;
	  else if (submit_status == submit_driver_FAIL)
	    break;
	}
	queue_index++;
      }
    }

    for (queue_index = 0; queue_index < ecl_queue_get_active_size(queue); queue_index++) {
      ecl_queue_node_type * node = queue->jobs[queue_index];
      switch (ecl_queue_node_get_status(node)) {
      case(ecl_queue_done):
	if (node->target_file != NULL) {
	  if (util_file_exists(node->target_file)) {
	    util_block_growing_file(node->target_file);   /* An attempt to ensure that all the ECLIPSE files are completly written */
	    util_block_growing_directory(node->run_path); 
	    ecl_queue_change_node_status(queue , node , ecl_queue_complete_OK);
	  } else {
	    bool verbose = true;
	    if (verbose) {
	      printf("Restarting: %s \n",node->ecl_base);
	    }
	    ecl_queue_change_node_status(queue , node , ecl_queue_waiting);
	    queue->status_list[ecl_queue_restart]++;
	  }
	} else 
	  ecl_queue_change_node_status(queue , node , ecl_queue_complete_OK);
	ecl_queue_free_job(queue , node);
	break;
      case(ecl_queue_exit):
	queue->status_list[ecl_queue_restart]++;
	ecl_queue_change_node_status(queue , node , ecl_queue_waiting);
	ecl_queue_free_job(queue , node);
	break;
      default:
	break;
      }
    }
    sleep(queue->sleep_time);
  } while ( (queue->status_list[ecl_queue_complete_OK] + queue->status_list[ecl_queue_complete_FAIL]) < num_total_run);
  printf("\n");
}



void ecl_queue_add_job(ecl_queue_type * queue , int external_id, int target_report) {
  pthread_mutex_lock( &queue->active_mutex );
  {
    int active_size  = queue->active_size;
    if (active_size == queue->size) 
      util_abort("%s: queue is already filled up with %d jobs - aborting \n",__func__ , queue->size);
    
    ecl_queue_initialize_node(queue , active_size , external_id , target_report);
    queue->active_size++;
  }
  pthread_mutex_unlock( &queue->active_mutex );
}


ecl_queue_type * ecl_queue_alloc(int size , int max_running , int max_submit , 
				 const char    	     * submit_cmd      	 , 
				 const char    	     * eclipse_exe     	 , 
				 const char    	     * eclipse_LD_path 	 , 
				 const char    	     * eclipse_config  	 ,
				 const char    	     * license_server  	 ,
				 const path_fmt_type * run_path_fmt 	 , 
				 const path_fmt_type * ecl_base_fmt 	 , 
				 const path_fmt_type * target_file_fmt ,
				 void * driver) {
  ecl_queue_type * queue = util_malloc(sizeof * queue , __func__);

  queue->sleep_time      = 1;
  queue->max_running     = max_running;
  queue->max_submit      = max_submit;
  queue->size            = size;
  queue->submit_cmd      = util_alloc_string_copy(submit_cmd);
  queue->eclipse_exe     = util_alloc_joined_string((const char *[3]) {"\"" , eclipse_exe , "\""} , 3 , "");
  queue->eclipse_config  = util_alloc_string_copy(eclipse_config);
  queue->license_server  = util_alloc_string_copy(license_server);
  if (eclipse_LD_path != NULL)
    queue->eclipse_LD_path = util_alloc_joined_string((const char *[3]) {"\"" , eclipse_LD_path , "\""} , 3 , "");
  else
    queue->eclipse_LD_path = NULL;

  queue->jobs            = util_malloc(size * sizeof * queue->jobs , __func__);
  {
    int i;
    for (i=0; i < size; i++) 
      queue->jobs[i] = ecl_queue_node_alloc();

    for (i=0; i < ecl_queue_max_state; i++)
      queue->status_list[i] = 0;
    queue->status_list[ecl_queue_null] = size;
  }

  queue->run_path_fmt = path_fmt_copyc(run_path_fmt);
  queue->ecl_base_fmt = path_fmt_copyc(ecl_base_fmt);
  if (target_file_fmt != NULL)
    queue->target_file_fmt = path_fmt_copyc(target_file_fmt);
  else
    queue->target_file_fmt = NULL;

  queue->driver = driver;
  basic_queue_driver_assert_cast(queue->driver);
  pthread_mutex_init( &queue->status_mutex , NULL );
  pthread_mutex_init( &queue->active_mutex , NULL );
  queue->active_size = 0;
  return queue;
}







void ecl_queue_free(ecl_queue_type * queue) {
  free(queue->submit_cmd);
  free(queue->eclipse_exe);
  free(queue->eclipse_config);
  free(queue->license_server);
  if (queue->eclipse_LD_path != NULL) free(queue->eclipse_LD_path);
  path_fmt_free(queue->run_path_fmt);
  path_fmt_free(queue->ecl_base_fmt);
  if (queue->target_file_fmt != NULL)
    path_fmt_free(queue->target_file_fmt);
  {
    int i;
    for (i=0; i < queue->size; i++) 
      ecl_queue_node_free(queue->jobs[i]);
    free(queue->jobs);
  }
  {
    basic_queue_driver_type * driver = queue->driver;
    driver->free_driver(driver);
  }
  free(queue);
  queue = NULL;
}
