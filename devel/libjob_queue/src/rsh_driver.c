#include <stdlib.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <netdb.h>
#include <basic_queue_driver.h>
#include <rsh_driver.h>
#include <util.h>
#include <pthread.h>
#include <arg_pack.h>
#include <errno.h>


struct rsh_job_struct {
  UTIL_TYPE_ID_DECLARATION;
  int          node_index;
  bool         active;       /* Means that it allocated - not really in use */ 
  job_status_type status;        
  pthread_t    run_thread;
  const char * host_name;    /* Currently not set */
  char       * run_path;
};



typedef struct {
  char 	     	  * host_name;
  int  	     	    max_running;
  int  	     	    running;
  pthread_mutex_t   host_mutex;
} rsh_host_type;



#define RSH_DRIVER_TYPE_ID 44963256
#define RSH_JOB_TYPE_ID    63256701


struct rsh_driver_struct {
  UTIL_TYPE_ID_DECLARATION
  QUEUE_DRIVER_FUNCTIONS
  pthread_mutex_t     submit_lock;
  pthread_attr_t      thread_attr;
  char              * rsh_command;
  int                 num_hosts;
  int                 last_host_index;
  rsh_host_type     **host_list;
};



/******************************************************************/
static UTIL_SAFE_CAST_FUNCTION( rsh_driver , RSH_DRIVER_TYPE_ID )
static UTIL_SAFE_CAST_FUNCTION( rsh_job , RSH_JOB_TYPE_ID )



/**
   If the host is for some reason not available, NULL should be returned.
*/

static rsh_host_type * rsh_host_alloc(const char * host_name , int max_running) {
  struct addrinfo * result;
  if (getaddrinfo(host_name , NULL , NULL , &result) == 0) {
    rsh_host_type * host = util_malloc(sizeof * host , __func__);
    
    host->host_name   = util_alloc_string_copy(host_name);
    host->max_running = max_running;
    host->running     = 0;
    pthread_mutex_init( &host->host_mutex , NULL );

    freeaddrinfo( result );
    return host;
  } else {
    fprintf(stderr,"** Warning: could not locate server: %s \n",host_name);
    return NULL;
  }
}


/*
  static void rsh_host_reset(rsh_host_type * rsh_host) {
  rsh_host->running = 0;
}
*/

static void rsh_host_free(rsh_host_type * rsh_host) {
  free(rsh_host->host_name);
  free(rsh_host);
}


static bool rsh_host_available(rsh_host_type * rsh_host) {
  bool available;

  pthread_mutex_lock( &rsh_host->host_mutex );
  if ((rsh_host->max_running - rsh_host->running) > 0) {
    available = true;
    rsh_host->running++;
  } else
    available = false;
  pthread_mutex_unlock( &rsh_host->host_mutex );

  return available;
}





static void rsh_host_submit_job(rsh_host_type * rsh_host , rsh_job_type * job, const char * rsh_cmd , const char * submit_cmd , const char * run_path) {
  /* 
     Observe that this job has already been added to the running jobs
     in the rsh_host_available function.
  */
  
  util_fork_exec(rsh_cmd , 3 , (const char *[3]) {rsh_host->host_name , submit_cmd , run_path} , true , NULL , NULL , NULL , NULL , NULL);
  job->status = JOB_QUEUE_DONE;

  pthread_mutex_lock( &rsh_host->host_mutex );
  rsh_host->running--;
  pthread_mutex_unlock( &rsh_host->host_mutex );  
}


/*
  static const char * rsh_host_get_hostname(const rsh_host_type * host) { return host->host_name; }
*/



static void * rsh_host_submit_job__(void * __arg_pack) {
  arg_pack_type * arg_pack = arg_pack_safe_cast(__arg_pack);
  char * rsh_cmd 	   = arg_pack_iget_ptr(arg_pack , 0); 
  rsh_host_type * rsh_host = arg_pack_iget_ptr(arg_pack , 1);
  char * submit_cmd 	   = arg_pack_iget_ptr(arg_pack , 2); 
  char * run_path          = arg_pack_iget_ptr(arg_pack , 3); 
  rsh_job_type * job       = arg_pack_iget_ptr(arg_pack , 4);

  rsh_host_submit_job(rsh_host , job , rsh_cmd , submit_cmd , run_path);
  pthread_exit( NULL );
  arg_pack_free( arg_pack );
}



  

/*****************************************************************/


/*****************************************************************/




rsh_job_type * rsh_job_alloc(int node_index , const char * run_path) {
  rsh_job_type * job;
  job = util_malloc(sizeof * job , __func__);
  job->active     = false;
  job->status     = JOB_QUEUE_WAITING;
  job->run_path   = util_alloc_string_copy(run_path);
  job->node_index = node_index;
  UTIL_TYPE_ID_INIT( job , RSH_JOB_TYPE_ID );
  return job;
}



void rsh_job_free(rsh_job_type * job) {
  free(job->run_path);
  free(job);
}




job_status_type rsh_driver_get_job_status(void * __driver , void * __job) {
  if (__job == NULL) 
    /* The job has not been registered at all ... */
    return JOB_QUEUE_NOT_ACTIVE;
  else {
    rsh_job_type    * job    = rsh_job_safe_cast( __job );
    {
      if (job->active == false) {
	util_abort("%s: internal error - should not query status on inactive jobs \n" , __func__);
	return JOB_QUEUE_NOT_ACTIVE;   /* Dummy to shut up compiler */
      } else 
	return job->status;
    }
  }
}



void rsh_driver_free_job(void * __driver , void * __job) {
  rsh_job_type    * job    = rsh_job_safe_cast( __job );
  rsh_job_free(job);
}



void rsh_driver_kill_job(void * __driver ,void  * __job) {
  rsh_job_type    * job    = rsh_job_safe_cast( __job );
  if (job->active)
    pthread_cancel( job->run_thread );
  rsh_job_free( job );
}



void * rsh_driver_submit_job(void  * __driver, 
                             int   node_index , 
                             const char * submit_cmd  	  , 
                             const char * run_path    	  ,
                             const char * job_name        ,
                             const char ** arg_list ) {
  
  rsh_driver_type * driver = rsh_driver_safe_cast( __driver );
  rsh_job_type  * job      = NULL; 
  {
    /* 
       command is freed in the start_routine() function
    */
    rsh_host_type * host = NULL;
    int ihost;
    int host_index = 0;
    pthread_mutex_lock( &driver->submit_lock );
    for (ihost = 0; ihost < driver->num_hosts; ihost++) {
      host_index = (ihost + driver->last_host_index) % driver->num_hosts;
      if (rsh_host_available(driver->host_list[host_index])) {
	host = driver->host_list[host_index];
	break;
      } 
    }
    driver->last_host_index = (host_index + 1) % driver->num_hosts;
    
    if (host != NULL) {
      /* A host is available */
      arg_pack_type * arg_pack = arg_pack_alloc();   /* The arg_pack is freed() in the rsh_host_submit_job__() function.
                                                        freeing it here is dangerous, because we might free it before the 
                                                        thread-called function is finished with it. */

      job = rsh_job_alloc(node_index , run_path);
      arg_pack_append_ptr(arg_pack ,  driver->rsh_command);
      arg_pack_append_ptr(arg_pack ,  host);
      arg_pack_append_ptr(arg_pack , (char *) submit_cmd);
      arg_pack_append_ptr(arg_pack , (char *) run_path);
      arg_pack_append_ptr(arg_pack , job);  
      
      {
	int pthread_return_value = pthread_create( &job->run_thread , &driver->thread_attr , rsh_host_submit_job__ , arg_pack);
	if (pthread_return_value != 0) 
	  util_abort("%s failed to create thread ERROR:%d  \n", __func__ , pthread_return_value);
      }
      job->status = JOB_QUEUE_RUNNING; 
      job->active = true;
    } 
    pthread_mutex_unlock( &driver->submit_lock );
  }
  return job;
}


void rsh_driver_free(rsh_driver_type * driver) {
  int ihost;
  for (ihost =0; ihost < driver->num_hosts; ihost++) 
    rsh_host_free(driver->host_list[ihost]);
  free(driver->host_list);

  pthread_attr_destroy ( &driver->thread_attr );
  free(driver);
  driver = NULL;
}


void rsh_driver_free__(void * __driver) {
  rsh_driver_type * driver = rsh_driver_safe_cast( __driver );
  rsh_driver_free( driver );
}




/**
*/

void * rsh_driver_alloc(const char * rsh_command, const hash_type * rsh_host_list) {
  rsh_driver_type * rsh_driver = util_malloc(sizeof * rsh_driver , __func__);
  UTIL_TYPE_ID_INIT( rsh_driver , RSH_DRIVER_TYPE_ID );
  pthread_mutex_init( &rsh_driver->submit_lock , NULL );
  pthread_attr_init( &rsh_driver->thread_attr );
  pthread_attr_setdetachstate( &rsh_driver->thread_attr , PTHREAD_CREATE_DETACHED );

  rsh_driver->rsh_command     	   = util_alloc_string_copy(rsh_command);
  rsh_driver->submit          	   = rsh_driver_submit_job;
  rsh_driver->get_status      	   = rsh_driver_get_job_status;
  rsh_driver->kill_job         	   = rsh_driver_kill_job;
  rsh_driver->free_job        	   = rsh_driver_free_job;
  rsh_driver->free_driver     	   = rsh_driver_free__;
  rsh_driver->display_info         = NULL;
  rsh_driver->driver_type          = RSH_DRIVER; 

  rsh_driver->num_hosts       	   = 0;
  rsh_driver->host_list       	   = NULL;
  rsh_driver->last_host_index 	   = 0;  
  {
    hash_iter_type * hash_iter = hash_iter_alloc( rsh_host_list );
    while (!hash_iter_is_complete( hash_iter )) {
      const char * host = hash_iter_get_next_key( hash_iter );
      int max_running   = hash_get_int( rsh_host_list , host );
      rsh_driver_add_host(rsh_driver , host , max_running);
    }
  }
  if (rsh_driver->num_hosts == 0) 
    util_abort("%s: failed to add any valid RSH hosts - aborting.\n",__func__);

  return rsh_driver;
}



void rsh_driver_add_host(rsh_driver_type * rsh_driver , const char * hostname , int host_max_running) {
  rsh_host_type * new_host = rsh_host_alloc(hostname , host_max_running);
  if (new_host != NULL) {
    rsh_driver->num_hosts++;
    rsh_driver->host_list = util_realloc(rsh_driver->host_list , rsh_driver->num_hosts * sizeof * rsh_driver->host_list , __func__);
    rsh_driver->host_list[(rsh_driver->num_hosts - 1)] = new_host;
  }
}




#undef RSH_JOB_ID    

/*****************************************************************/

