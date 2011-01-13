#include <stdlib.h>
#include <stdbool.h>
#include <queue_driver.h>
#include <lsf_driver.h>
#include <local_driver.h>
#include <rsh_driver.h>



struct queue_driver_struct {
  /* 
     Function pointers - pointing to low level functions in the implementations of
     e.g. lsf_driver.
  */
  submit_job_ftype           * submit;             
  free_job_ftype             * free_job;           
  kill_job_ftype             * kill_job;           
  get_status_ftype           * get_status;         
  free_queue_driver_ftype    * free_driver;        
  set_option_ftype           * set_option;
  get_option_ftype           * get_option;
  has_option_ftype           * has_option; 
  
  
  /*
    Data - common to all driver types.
  */
  char             * name;           /* String name of driver. */
  job_driver_type    driver_type;    /* Enum value for driver. */
  int                max_running;    /* Possible to maintain different max_running values for different drivers. */ 
  void             * data;           /* Driver specific data - passed as first argument to the driver functions above. */
};



/**
   Observe that after the driver instance has been allocated it does
   NOT support modification of the common fields, only the data owned
   by the specific low level driver, i.e. the LSF data, can be
   modified runtime.

   The driver returned from the queue_driver_alloc_empty() function is
   NOT properly initialized and NOT ready for use.  
*/

static queue_driver_type * queue_driver_alloc_empty( ) {
  queue_driver_type * driver = util_malloc( sizeof * driver, __func__);
  driver->max_running = 0;
  driver->driver_type = NULL_DRIVER;
  driver->submit      = NULL;
  driver->get_status  = NULL;
  driver->kill_job    = NULL;
  driver->free_job    = NULL;
  driver->free_driver = NULL;
  driver->get_option  = NULL;
  driver->set_option  = NULL;
  driver->has_option  = NULL;
  driver->name        = NULL;
  driver->data        = NULL;         
  
  return driver;
}

/*****************************************************************/
/* Set option */
void queue_driver_set_option( queue_driver_type * driver , const char * option_key , const void * value) {
  if (driver->set_option != NULL) 
    /* The actual low level set functions can not fail! */
    driver->set_option( driver->data , option_key , value );
  else
    util_abort("%s: driver:%s does not support run time setting of options\n",__func__ , driver->name );
}

void queue_driver_set_int_option( queue_driver_type * driver , const char * option_key , int int_value) {
  const void * void_value = &int_value;
  queue_driver_set_option( driver , option_key , void_value );
}

void queue_driver_set_string_option( queue_driver_type * driver , const char * option_key , const char * value) {
  queue_driver_set_option( driver , option_key , value);
}

/*****************************************************************/

bool queue_driver_has_option( queue_driver_type * driver , const char * option_key ) {
  if (driver->has_option != NULL)
    return driver->has_option( driver , option_key );
  else
    return false;
}

/*****************************************************************/

const void * queue_driver_get_option( queue_driver_type * driver , const char * option_key ) {
  if (driver->get_option != NULL) 
    /* The actual low level set functions can not fail! */
    return driver->get_option( driver->data , option_key );
  else {
    util_abort("%s: driver:%s does not support run time reading of options\n",__func__ , driver->name );
    return NULL;
  }
}

/*****************************************************************/


queue_driver_type * queue_driver_alloc_LSF(const char * queue_name , const char * resource_request , const char * remote_lsf_server , int num_cpu) {
  queue_driver_type * driver = queue_driver_alloc_empty( );

  driver->driver_type  = LSF_DRIVER;
  driver->submit      = lsf_driver_submit_job;
  driver->get_status  = lsf_driver_get_job_status;
  driver->kill_job    = lsf_driver_kill_job;
  driver->free_job    = lsf_driver_free_job;
  driver->free_driver = lsf_driver_free__;
  driver->set_option  = lsf_driver_set_option;
  driver->get_option  = lsf_driver_get_option;
  driver->has_option  = lsf_driver_has_option;
  driver->name        = util_alloc_string_copy("LSF");
  driver->data        = lsf_driver_alloc( );

  queue_driver_set_option( driver , LSF_QUEUE    , queue_name );
  queue_driver_set_option( driver , LSF_RESOURCE , resource_request );
  queue_driver_set_option( driver , LSF_SERVER   , remote_lsf_server );
  queue_driver_set_int_option( driver , LSF_NUM_CPU , num_cpu );

  return driver;
}


queue_driver_type * queue_driver_alloc_RSH( const char * rsh_cmd , const hash_type * rsh_hostlist) {
  queue_driver_type * driver = queue_driver_alloc_empty( );

  driver->driver_type = RSH_DRIVER;
  driver->submit      = rsh_driver_submit_job;
  driver->get_status  = rsh_driver_get_job_status;
  driver->kill_job    = rsh_driver_kill_job;
  driver->free_job    = rsh_driver_free_job;
  driver->free_driver = rsh_driver_free__;
  driver->set_option  = rsh_driver_set_option; 
  driver->get_option  = rsh_driver_get_option; 
  driver->name        = util_alloc_string_copy("RSH");
  driver->data        = rsh_driver_alloc( );
  
  queue_driver_set_option( driver , RSH_HOSTLIST , rsh_hostlist );
  queue_driver_set_option( driver , RSH_CMD      , rsh_cmd );
  
  return driver;
}


queue_driver_type * queue_driver_alloc_local( ) {
  queue_driver_type * driver = queue_driver_alloc_empty( );

  driver->driver_type = LOCAL_DRIVER;
  driver->submit      = local_driver_submit_job;
  driver->get_status  = local_driver_get_job_status;
  driver->kill_job    = local_driver_kill_job;
  driver->free_job    = local_driver_free_job;
  driver->free_driver = local_driver_free__;
  driver->name        = util_alloc_string_copy("local");
  driver->data        = local_driver_alloc( );
  /* No options set for the local driver. */

  return driver;
}
/*****************************************************************/

void queue_driver_set_max_running( queue_driver_type * driver , int max_running) {
  driver->max_running = max_running;
}

int queue_driver_get_max_running( const queue_driver_type * driver ) {
  return driver->max_running;
}

const char * queue_driver_get_name( const queue_driver_type * driver ) {
  return driver->name;
}


/*****************************************************************/
/* These are the functions used by the job_queue layer. */

void * queue_driver_submit_job( queue_driver_type * driver, const char * run_cmd , const char * run_path , const char * job_name , int argc , const char ** argv) {
  return driver->submit( driver->data , run_cmd , run_path , job_name , argc , argv );
}

void queue_driver_free_job( queue_driver_type * driver , void * job_data ) {
  driver->free_job( job_data );
}

void queue_driver_kill_job( queue_driver_type * driver , void * job_data ) {
  driver->kill_job( driver->data , job_data );
}

job_status_type  queue_driver_get_status( queue_driver_type * driver , void * job_data) {
  return driver->get_status( driver->data , job_data );
}

void queue_driver_free_driver( queue_driver_type * driver ) {
  driver->free_driver( driver->data );
}

/*****************************************************************/

void queue_driver_free( queue_driver_type * driver ) {
  queue_driver_free_driver( driver );
  util_safe_free( driver->name );
  free( driver );
}
