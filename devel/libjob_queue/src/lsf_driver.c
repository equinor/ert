#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <basic_queue_driver.h>
#include <lsf_driver.h>
#include <util.h>
#include <hash.h>
#include <pthread.h>
#include <stringlist.h>

/**
   the LSF_LIBRARY_DRIVER uses the lsb/lsf libraries directly; if the
   librarie/headers .... are not available for linking we can use the
   lsf_system_driver instead - the latter is based on calling bsub and
   bjobs with system calls, using temporary files and parsing the
   output; the former is of course highly preferred.

   Documentation/examples of programming towards the lsf libraries can
   be found in /prog/LSF/7.0/misc/examples
*/


/* 
   Check the #defines - one-and-only-one - of the symbols
   LSF_LIBRARY_DRIVER and LSF_SYSTEM_DRIVER must be defined.
*/

#ifdef LSF_LIBRARY_DRIVER
#ifdef LSF_SYSTEM_DRIVER
#error: Both symbols LSF_LIBRARY_DRIVER and LSF_SYSTEM_DRIVER are defined; invalid configuration.
#endif
#endif

#ifndef  LSF_LIBRARY_DRIVER
#ifndef  LSF_SYSTEM_DRIVER
#error: Neither LSF_LIBRARY_DRIVER nor LSF_SYSTEM_DRIVER are defined; invalid configuration.
#endif
#endif

/*****************************************************************/

#ifdef  LSF_LIBRARY_DRIVER
#include <lsf/lsbatch.h>
#endif


#define LSF_DRIVER_TYPE_ID 10078365
#define LSF_JOB_TYPE_ID    99639007

struct lsf_job_struct {
  UTIL_TYPE_ID_DECLARATION;
  long int  lsf_jobnr;
  int       num_exec_host;
  char    **exec_host;
#ifdef LSF_SYSTEM_DRIVER
  char     * lsf_jobnr_char;  /* Used to look up the job status in the bjobs_cache hash table */
#endif
};

#define BJOBS_REFRESH_TIME 5

struct lsf_driver_struct {
  UTIL_TYPE_ID_DECLARATION;
  QUEUE_DRIVER_FUNCTIONS
  char             * queue_name;
  char             * resource_request;
  pthread_mutex_t    submit_lock;
  int                 num_cpu;
#ifdef LSF_LIBRARY_DRIVER
  struct submit      lsf_request;
  struct submitReply lsf_reply; 
#else
  time_t              last_bjobs_update;
  hash_type         * my_jobs;     /* A hash table of all jobs submitted by this ERT instance - to ensure that we do not check status of old jobs in e.g. ZOMBIE status. */  
  hash_type         * status_map;
  hash_type         * bjobs_cache;     /* The output of calling bjobs is cached in this table. */
  pthread_mutex_t     bjobs_mutex;     /* Only one thread should update the bjobs_chache table. */
  char              * bjobs_executable;
  char              * bsub_executable;
  char              * bkill_executable;
#endif
};



/*****************************************************************/

UTIL_SAFE_CAST_FUNCTION( lsf_driver , LSF_DRIVER_TYPE_ID)
static UTIL_SAFE_CAST_FUNCTION_CONST( lsf_driver , LSF_DRIVER_TYPE_ID)
static UTIL_SAFE_CAST_FUNCTION( lsf_job , LSF_JOB_TYPE_ID)

lsf_job_type * lsf_job_alloc() {
  lsf_job_type * job;
  job = util_malloc(sizeof * job , __func__);
  job->num_exec_host = 0;
  job->exec_host     = NULL;

#ifdef LSF_SYSTEM_DRIVER
  job->lsf_jobnr_char = NULL;
#endif
  UTIL_TYPE_ID_INIT( job , LSF_JOB_TYPE_ID);
  return job;
}



void lsf_job_free(lsf_job_type * job) {
#ifdef LSF_SYSTEM_DRIVER
  util_safe_free(job->lsf_jobnr_char);
#endif
  util_free_stringlist(job->exec_host , job->num_exec_host);
  free(job);
}


#ifdef LSF_SYSTEM_DRIVER
static int lsf_job_parse_bsub_stdout(const char * stdout_file) {
  int     jobid = -1;
  FILE * stream = util_fopen(stdout_file , "r");
  if (util_fseek_string(stream , "<" , true , true)) {
    char * jobid_string = util_fscanf_alloc_upto(stream , ">" , false);
    if (jobid_string != NULL) {
      jobid = atoi( jobid_string );
      free( jobid_string );
    } else
      util_abort("%s: Could not extract job id from bsub submit_file:%s \n",__func__ , stdout_file );
  } else
    util_abort("%s: Could not extract job id from bsub submit_file:%s \n",__func__ , stdout_file );
  
  fclose( stream );
  return jobid;
}




/*
  Essential to source /prog/LSF/conf/cshrc.lsf before invoking lf
  functions on xStatoil.
*/


static int lsf_driver_submit_system_job(lsf_driver_type * driver , 
                                        const char *  run_path , 
                                        const char *  job_name , 
                                        const char *  lsf_queue , 
                                        const char *  resource_request , 
                                        const char *  submit_cmd,
                                        int           job_argc,
                                        const char ** job_argv) {
  int argc;
  char ** argv;
  int job_id;
  char * tmp_file         = util_alloc_tmp_file("/tmp" , "enkf-submit" , true);
  char * lsf_stdout       = util_alloc_filename(run_path , job_name , "LSF-stdout");

  char num_cpu_string[4];
  sprintf(num_cpu_string , "%d" , driver->num_cpu);

  if (resource_request != NULL) 
    argc = job_argc + 11;
  else
    argc = job_argc + 9;

  argv = util_malloc( argc * sizeof * argv , __func__ ); /* This argv structure only contains pointers to string storage;
                                                            does not alloacte anything to itself. */

  {
    int offset = 8;
    argv[0] = "-o";  
    argv[1] = lsf_stdout;
    argv[2] = "-q";  
    argv[3] = lsf_queue;
    argv[4] = "-J";  
    argv[5] = job_name;
    argv[6] = "-n";  
    argv[7] = num_cpu_string;
    if (resource_request != NULL) {
      argv[8] = "-R";
      argv[9] = resource_request;
      offset  = 10;
    }
    argv[offset] = submit_cmd;
    {
      int iarg;
      for (iarg = 0; iarg < argc; iarg++)
        argv[iarg + offset + 1] = job_argv[ iarg ];
    }
  }
  
  util_fork_exec(driver->bsub_executable , 12 , argv , true , NULL , NULL , NULL , tmp_file , NULL);
  
  job_id = lsf_job_parse_bsub_stdout(tmp_file);
  util_unlink_existing(tmp_file); 
  free(lsf_stdout);
  free(tmp_file);
  free(argv);
  return job_id;
}



static int lsf_driver_get_status__(lsf_driver_type * driver , const char * status, const char * job_id) {

  if (hash_has_key( driver->status_map , status))
    return hash_get_int( driver->status_map , status);
  else 
    util_exit("The lsf_status:%s  for job:%s is not recognized; call your LSF administrator - sorry :-( \n", status , job_id);
}



static void lsf_driver_update_bjobs_table(lsf_driver_type * driver) {
  char * tmp_file   = util_alloc_tmp_file("/tmp" , "enkf-bjobs" , true);
  util_fork_exec(driver->bjobs_executable , 1 , (const char *[1]) {"-a"} , true , NULL , NULL , NULL , tmp_file , NULL);
  {
    char user[32];
    char status[16];
    FILE *stream = util_fopen(tmp_file , "r");;
    bool at_eof = false;
    hash_clear(driver->bjobs_cache);
    util_fskip_lines(stream , 1);
    while (!at_eof) {
      char * line = util_fscanf_alloc_line(stream , &at_eof);
      if (line != NULL) {
	int  job_id_int;

	if (sscanf(line , "%d %s %s", &job_id_int , user , status) == 3) {
	  char * job_id = util_alloc_sprintf("%d" , job_id_int);

          if (hash_has_key( driver->my_jobs , job_id ))   /* Consider only jobs submitted by this ERT instance - not old jobs lying around from the same user. */
            hash_insert_int(driver->bjobs_cache , job_id , lsf_driver_get_status__( driver , status , job_id));
          else
            printf("%s: skipping job:%s \n",__func__ , job_id );
	  free(job_id);
	}
	free(line);
      }
    }
    fclose(stream);
  }
  util_unlink_existing(tmp_file); 
  free(tmp_file);
}
#endif



#ifdef LSF_LIBRARY_DRIVER
#define CASE_SET(s1,s2) case(s1):  status = s2; break;
static job_status_type lsf_driver_get_job_status_libary(void * __driver , void * __job) {
  if (__job == NULL) 
    /* the job has not been registered at all ... */
    return JOB_QUEUE_NOT_ACTIVE;
  else {
    lsf_job_type    * job    = lsf_job_safe_cast( __job );
    {
      job_status_type status;
      struct jobInfoEnt *job_info;
      if (lsb_openjobinfo(job->lsf_jobnr , NULL , NULL , NULL , NULL , ALL_JOB) != 1) {
	/* 
	   Failed to get information about the job - we boldly assume
	   the following situation has occured:
           
             1. The job is running happily along.
	     2. The lsf deamon is not responding for a long time.
	     3. The job finishes, and is eventually expired from the LSF job database.
	     4. The lsf deamon answers again - but can not find the job...
	     
	*/
	fprintf(stderr,"Warning: failed to get status information for job:%ld - assuming it is finished. \n", job->lsf_jobnr);
	status = JOB_QUEUE_DONE;
      } else {
	job_info = lsb_readjobinfo( NULL );
	lsb_closejobinfo();
	if (job->num_exec_host == 0) {
	  job->num_exec_host = job_info->numExHosts;
	  job->exec_host     = util_alloc_stringlist_copy( (const char **) job_info->exHosts , job->num_exec_host);
	}
	
	switch (job_info->status) {
	  CASE_SET(JOB_STAT_PEND  , JOB_QUEUE_PENDING);
	  CASE_SET(JOB_STAT_SSUSP , JOB_QUEUE_RUNNING);
	  CASE_SET(JOB_STAT_RUN   , JOB_QUEUE_RUNNING);
	  CASE_SET(JOB_STAT_EXIT  , JOB_QUEUE_EXIT);
	  CASE_SET(JOB_STAT_DONE  , JOB_QUEUE_DONE);
	  CASE_SET(JOB_STAT_PDONE , JOB_QUEUE_DONE);
	  CASE_SET(JOB_STAT_PERR  , JOB_QUEUE_EXIT);
	  CASE_SET(192            , JOB_QUEUE_DONE); /* this 192 seems to pop up - where the fuck 
                                                        does it come frome ??  _pdone + _ususp ??? */
	default:
          util_abort("%s: job:%ld lsf_status:%d not recognized - internal LSF fuck up - aborting \n",__func__ , job->lsf_jobnr , job_info->status);
	  status = JOB_QUEUE_DONE; /* ????  */
	}
      }
      
      return status;
    }
  }
}
#undef case

#else

static job_status_type lsf_driver_get_job_status_system(void * __driver , void * __job) {
  job_status_type status = JOB_QUEUE_NOT_ACTIVE;
  
  if (__job != NULL) {
    lsf_job_type    * job    = lsf_job_safe_cast( __job );
    lsf_driver_type * driver = lsf_driver_safe_cast( __driver );
    
    {
      /**
         Updating the bjobs_table of the driver involves a significant change in
         the internal state of the driver; that is semantically a bit
         unfortunate because this is clearly a get() function; to protect
         against concurrent updates of this table we use a mutex.
      */
      pthread_mutex_lock( &driver->bjobs_mutex );
      {
        if (difftime(time(NULL) , driver->last_bjobs_update) > BJOBS_REFRESH_TIME) {
          lsf_driver_update_bjobs_table(driver);
          driver->last_bjobs_update = time( NULL );
        }
      }
      pthread_mutex_unlock( &driver->bjobs_mutex );
      
      
      if (hash_has_key( driver->bjobs_cache , job->lsf_jobnr_char) ) 
	status = hash_get_int(driver->bjobs_cache , job->lsf_jobnr_char);
      else
	/* 
	   It might be running - but since job != NULL it is at least in the queue system.
	*/
	status = JOB_QUEUE_PENDING;

    }
  }
  
  return status;
}

#endif



job_status_type lsf_driver_get_job_status(void * __driver , void * __job) {
  job_status_type status;
#ifdef LSF_LIBRARY_DRIVER
  status = lsf_driver_get_job_status_libary(__driver , __job);
#else
  status = lsf_driver_get_job_status_system(__driver , __job);
#endif
  return status;
}


void lsf_driver_display_info( void * __driver , void * __job) {
  lsf_job_type    * job    = lsf_job_safe_cast( __job );
  printf("Executing host: ");
  {
    int i;
    for (i=0; i < job->num_exec_host; i++)
      printf("%s ", job->exec_host[i]);
  }
}



void lsf_driver_free_job(void * __job) {
  lsf_job_type    * job    = lsf_job_safe_cast( __job );
  lsf_job_free(job);
}


int lsf_driver_get_num_cpu( const void * __lsf_driver ) {
  const lsf_driver_type * lsf_driver = lsf_driver_safe_cast_const( __lsf_driver );
  return lsf_driver->num_cpu;
}

void lsf_driver_set_num_cpu( void * __lsf_driver , int num_cpu) {
  lsf_driver_type * lsf_driver = lsf_driver_safe_cast( __lsf_driver );
  lsf_driver->num_cpu = num_cpu;
}


void lsf_driver_kill_job(void * __driver , void * __job) {
  lsf_job_type    * job    = lsf_job_safe_cast( __job );
  {
#ifdef LSF_LIBRARY_DRIVER
    lsb_forcekilljob(job->lsf_jobnr);
#else 
    {
      lsf_driver_type * driver = lsf_driver_safe_cast( __driver );
      util_fork_exec(driver->bkill_executable , 1 , (const char **)  &job->lsf_jobnr_char , true , NULL , NULL , NULL , NULL , NULL);
    }
#endif
  }
  lsf_job_free( job );
}




void * lsf_driver_submit_job(void * __driver , 
                             const char  * submit_cmd  	  , 
                             const char  * run_path    	  , 
                             const char  * job_name ,
                             int           argc,     
                             const char ** argv ) {
  lsf_driver_type * driver = lsf_driver_safe_cast( __driver );
  printf("Hei - skal submitte en job .. \n");
  {
    lsf_job_type * job 		  = lsf_job_alloc();
    char * lsf_stdout  		  = util_alloc_joined_string( (const char *[2]) {run_path   , "/LSF.stdout"}  , 2 , "");
    pthread_mutex_lock( &driver->submit_lock );
#ifdef LSF_LIBRARY_DRIVER
    char * command;
    {
      buffer_type * command_buffer = buffer_alloc( 256 );
      buffer_fwrite_char_ptr( command_buffer , submit_cmd );
      for (int iarg = 0; iarg < argc; iarg++) {
        buffer_fwrite_char( command_buffer , ' ');
        buffer_fwrite_char_ptr( command_buffer , argv[ iarg ]);
      }
      buffer_terminate_char_ptr( command_buffer );
      command = buffer_get_data( command_buffer );
      buffer_free_container( command_buffer );
    }
    
    {
      int options = SUB_QUEUE + SUB_JOB_NAME + SUB_OUT_FILE;
      if (driver->resource_request != NULL) {
	options += SUB_RES_REQ;
	driver->lsf_request.resReq = driver->resource_request;
      }
      driver->lsf_request.options = options;
    }
    driver->lsf_request.jobName       = (char *) job_name;
    driver->lsf_request.outFile       = lsf_stdout;
    driver->lsf_request.command       = command;
    driver->lsf_request.numProcessors = driver->num_cpu;
    job->lsf_jobnr = lsb_submit( &driver->lsf_request , &driver->lsf_reply );
    printf("Job: \"%s\" submitted ??? \n" , command);
    free( command );  /* I trust the lsf layer is finished with the command? */
#else
    {
      char * quoted_resource_request = NULL;
      if (driver->resource_request != NULL)
	quoted_resource_request = util_alloc_sprintf("\"%s\"" , driver->resource_request);

      job->lsf_jobnr      = lsf_driver_submit_system_job( driver , run_path , job_name , driver->queue_name , quoted_resource_request , submit_cmd , argc, argv);
      job->lsf_jobnr_char = util_alloc_sprintf("%ld" , job->lsf_jobnr);
      hash_insert_ref( driver->my_jobs , job->lsf_jobnr_char , NULL );   
      util_safe_free(quoted_resource_request);
    }
#endif
    pthread_mutex_unlock( &driver->submit_lock );
    free(lsf_stdout);
    
    if (job->lsf_jobnr > 0) 
      return job;
    else {
      /*
	The submit failed - the queue system shall handle
	NULL return values.
      */
#ifdef LSF_LIBRARY_DRIVER
      fprintf(stderr,"%s: ** Warning: lsb_submit() failed: %s/%d \n",__func__ , lsb_sysmsg() , lsberrno);
#endif
      lsf_job_free(job);
      return NULL;
    }
  }
}



void lsf_driver_free(lsf_driver_type * driver ) {
  free(driver->queue_name);
  util_safe_free(driver->resource_request );
#ifdef LSF_SYSTEM_DRIVER
  hash_free(driver->status_map);
  hash_free(driver->bjobs_cache);
  hash_free(driver->my_jobs);
  free( driver->bjobs_executable );
  free( driver->bsub_executable );
  free( driver->bkill_executable );
#endif
  free(driver);
  driver = NULL;
}

void lsf_driver_free__(void * __driver ) {
  lsf_driver_type * driver = lsf_driver_safe_cast( __driver );
  lsf_driver_free( driver );
}



void lsf_driver_set_queue_name( lsf_driver_type * driver, const char * queue_name ) {
  driver->queue_name = util_realloc_string_copy( driver->queue_name , queue_name );
}


void lsf_driver_set_resource_request( lsf_driver_type * driver, const char * resource_request ) {
  driver->resource_request = util_realloc_string_copy( driver->resource_request , resource_request );
}





void * lsf_driver_alloc(const char * queue_name , const char * resource_request , int num_cpu) {
  lsf_driver_type * lsf_driver 	   = util_malloc(sizeof * lsf_driver , __func__);
  lsf_driver->queue_name           = NULL;
  lsf_driver->resource_request     = NULL;
  lsf_driver_set_queue_name( lsf_driver , queue_name );
  lsf_driver_set_resource_request( lsf_driver , resource_request );
  lsf_driver->queue_name       	   = util_alloc_string_copy(queue_name);
  UTIL_TYPE_ID_INIT( lsf_driver , LSF_DRIVER_TYPE_ID);
  lsf_driver->submit           	   = lsf_driver_submit_job;
  lsf_driver->get_status       	   = lsf_driver_get_job_status;
  lsf_driver->kill_job             = lsf_driver_kill_job;
  lsf_driver->free_job         	   = lsf_driver_free_job;
  lsf_driver->free_driver      	   = lsf_driver_free__;
  lsf_driver->driver_type          = LSF_DRIVER;
  lsf_driver->num_cpu              = num_cpu;
  pthread_mutex_init( &lsf_driver->submit_lock , NULL );

#ifdef LSF_LIBRARY_DRIVER
  memset(&lsf_driver->lsf_request , 0 , sizeof (lsf_driver->lsf_request));
  lsf_driver->lsf_request.queue     	   = lsf_driver->queue_name;
  lsf_driver->lsf_request.beginTime 	   = 0;
  lsf_driver->lsf_request.termTime   	   = 0;   
  lsf_driver->lsf_request.numProcessors    = 1;
  lsf_driver->lsf_request.maxNumProcessors = 1;
  {
    int i;
    for (i=0; i < LSF_RLIM_NLIMITS; i++) 
      lsf_driver->lsf_request.rLimits[i] = DEFAULT_RLIMIT;
  }
  lsf_driver->lsf_request.options2 = 0;
  if (lsb_init(NULL) != 0) 
    util_abort("%s failed to initialize LSF environment : %s/%d  \n",__func__ , lsb_sysmsg() , lsberrno);
  setenv("BSUB_QUIET" , "yes" , 1);
  lsf_driver->display_info         = lsf_driver_display_info;
#else
  lsf_driver->last_bjobs_update   = time( NULL );
  lsf_driver->bjobs_cache 	  = hash_alloc(); 
  lsf_driver->my_jobs 	          = hash_alloc(); 
  lsf_driver->status_map   	  = hash_alloc();
  lsf_driver->bjobs_executable    = util_alloc_PATH_executable( "bjobs" );
  lsf_driver->bsub_executable     = util_alloc_PATH_executable( "bsub" );
  lsf_driver->bkill_executable    = util_alloc_PATH_executable( "bkill" );
  lsf_driver->display_info        = NULL;                                      /* The system driver does not have any display info function. */
  hash_insert_int(lsf_driver->status_map , "PEND"   , JOB_QUEUE_PENDING);
  hash_insert_int(lsf_driver->status_map , "SSUSP"  , JOB_QUEUE_RUNNING);
  hash_insert_int(lsf_driver->status_map , "PSUSP"  , JOB_QUEUE_PENDING);
  hash_insert_int(lsf_driver->status_map , "RUN"    , JOB_QUEUE_RUNNING);
  hash_insert_int(lsf_driver->status_map , "EXIT"   , JOB_QUEUE_EXIT);
  hash_insert_int(lsf_driver->status_map , "USUSP"  , JOB_QUEUE_RUNNING);
  hash_insert_int(lsf_driver->status_map , "DONE"   , JOB_QUEUE_DONE);
  hash_insert_int(lsf_driver->status_map , "UNKWN"  , JOB_QUEUE_EXIT);    /* Uncertain about this one */
  pthread_mutex_init( &lsf_driver->bjobs_mutex , NULL );
#endif
  return lsf_driver;
}


/*****************************************************************/

