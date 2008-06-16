#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <basic_queue_driver.h>
#include <lsf_driver.h>
#include <util.h>
#include <pthread.h>
#include <void_arg.h>

/**
   the LSF_LIBRARY_DRIVER uses the lsb/lsf libraries directly; if the
   librarie/headers .... are not available for linking we can use the
   lsf_system_driver instead - the latter is based on calling bsub and
   bjobs with system calls, using temporary files and parsing the
   output.
*/


#ifdef i386
#define LSF_SYSTEM_DRIVER
#else
#ifdef USE_LSF_LIBRARY
#define LSF_LIBRARY_DRIVER
#include <lsf/lsbatch.h>
#else
#define LSF_SYSTEM_DRIVER
#endif
#endif




struct lsf_job_struct {
  int 	    __basic_id;
  int  	    __lsf_id;
  long int  lsf_jobnr;
#ifdef LSF_SYSTEM_DRIVER
  char    * lsf_jobnr_char;  /* Used to look up the job status in the bjobs_output hash table */
#endif
};




struct lsf_driver_struct {
  BASIC_QUEUE_DRIVER_FIELDS
  int __lsf_id;
  char * resource_request;
  char * queue_name;
  pthread_mutex_t    submit_lock;

#ifdef LSF_LIBRARY_DRIVER
  struct submit      lsf_request;
  struct submitReply lsf_reply; 
#else
  time_t             last_bjobs_update;
  hash_type         *status_map;
  hash_type         *bjobs_output;
  pthread_mutex_t    bjobs_lock;
#endif
};





/*****************************************************************/

#define LSF_DRIVER_ID  1001
#define LSF_JOB_ID     2001


void lsf_driver_assert_cast(const lsf_driver_type * queue_driver) {
  if (queue_driver->__lsf_id != LSF_DRIVER_ID) {
    fprintf(stderr,"%s: internal error - cast failed \n",__func__);
    abort();
  }
}


void lsf_driver_init(lsf_driver_type * queue_driver) {
  queue_driver->__lsf_id = LSF_DRIVER_ID;
}


void lsf_job_assert_cast(const lsf_job_type * queue_job) {
  if (queue_job->__lsf_id != LSF_JOB_ID) {
    fprintf(stderr,"%s: internal error - cast failed \n",__func__);
    abort();
  }
}



lsf_job_type * lsf_job_alloc() {
  lsf_job_type * job;
  job = util_malloc(sizeof * job , __func__);
  job->__lsf_id = LSF_JOB_ID;
#ifdef LSF_SYSTEM_DRIVER
  job->lsf_jobnr_char = NULL;
#endif
  return job;
}


void lsf_job_free(lsf_job_type * job) {
#ifdef LSF_SYSTEM_DRIVER
  util_safe_free(job->lsf_jobnr_char);
#endif
  free(job);
}


#ifdef LSF_SYSTEM_DRIVER
static int lsf_job_parse_bsub_stdout(const char * stdout_file) {
  int jobid;
  FILE * stream = util_fopen(stdout_file , "r");
  {
    char buffer[16];
    int c;
    int i;
    do {
      c = fgetc(stream);
    } while (c != '<');

    i = -1;
    do {
      i++;
      buffer[i] = fgetc(stream);
    } while(buffer[i] != '>');
    buffer[i] = '\0';
    jobid = atoi(buffer);
  }
  fclose(stream);
  return jobid;
}


static int lsf_driver_submit_system_job(const char * run_path , const char * job_name , const char * lsf_queue , const char * resource_request , const char * submit_cmd) {
  int job_id;
  char * tmp_file         = util_alloc_tmp_file("/tmp" , "enkf-submit" , true);
  char * lsf_stdout       = util_alloc_filename(run_path , job_name , "LSF-stdout");
  /*
    In total: 9 arguments
    bsub -o <lsf.stdout> -q <lsf_queue> -J <job_name> -R<"resource request"> <cmd> run_path 
  */
  util_vfork_exec("bsub" , 9 , (const char *[9]) {"-o" , lsf_stdout , "-q" , lsf_queue , "-J" , job_name , resource_request , submit_cmd , run_path} , true , NULL , NULL , tmp_file , NULL);

  job_id = lsf_job_parse_bsub_stdout(tmp_file);
  util_unlink_existing(tmp_file); 
  free(lsf_stdout);
  free(tmp_file);
  return job_id;
}




static void lsf_driver_update_bjobs_table(lsf_driver_type * driver) {
  char * tmp_file   = util_alloc_tmp_file("/tmp" , "enkf-bjobs" , true);
  util_vfork_exec("bjobs", 1 , (const char *[1]) {"-a"} , true , NULL , NULL , tmp_file , NULL);
  
  {
    int  job_id_int;
    char *job_id;
    char user[32];
    char status[16];
    FILE *stream = util_fopen(tmp_file , "r");;
    bool at_eof = false;
    hash_clear(driver->bjobs_output);
    util_fskip_lines(stream , 1);
    while (!at_eof) {
      char * line = util_fscanf_alloc_line(stream , &at_eof);
      if (line != NULL) {
	
	if (sscanf(line , "%d %s %s", &job_id_int , user , status) == 3) {
	  job_id = util_alloc_sprintf("%d" , job_id_int);
	  hash_insert_int(driver->bjobs_output , job_id , hash_get_int(driver->status_map , status));
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
#define case(s1,s2) case(s1):  status = s2; break;
static ecl_job_status_type lsf_driver_get_job_status_libary(basic_queue_driver_type * __driver , basic_queue_job_type * __job) {
  if (__job == NULL) 
    /* the job has not been registered at all ... */
    return job_queue_null;
  else {
    lsf_job_type    * job    = (lsf_job_type    *) __job;
    lsf_driver_type * driver = (lsf_driver_type *) __driver;
    lsf_driver_assert_cast(driver); 
    lsf_job_assert_cast(job);
    {
      ecl_job_status_type status;
      struct jobInfoEnt *job_info;
      if (lsb_openjobinfo(job->lsf_jobnr , NULL , NULL , NULL , NULL , ALL_JOB) != 1) {
	fprintf(stderr,"%s: failed to get information about lsf job:%ld - aborting \n",__func__ , job->lsf_jobnr);
	abort();
      }
      job_info = lsb_readjobinfo( NULL );
      lsb_closejobinfo();

      switch (job_info->status) {
	case(JOB_STAT_PEND  , job_queue_pending);
	case(JOB_STAT_SSUSP , job_queue_running);
	case(JOB_STAT_RUN   , job_queue_running);
	case(JOB_STAT_EXIT  , job_queue_exit);
	case(JOB_STAT_DONE  , job_queue_done);
	case(JOB_STAT_PDONE , job_queue_done);
	case(JOB_STAT_PERR  , job_queue_exit);
	case(192            , job_queue_done); /* this 192 seems to pop up - where the fuck it comes frome  _pdone + _ususp ??? */
      default:
	fprintf(stderr,"%s: job:%ld lsf_status:%d not handled - aborting \n",__func__ , job->lsf_jobnr , job_info->status);
	status = job_queue_done; /* ????  */
      }
      
      return status;
    }
  }
}
#undef case

#else

static ecl_job_status_type lsf_driver_get_job_status_system(basic_queue_driver_type * __driver , basic_queue_job_type * __job) {
  const int bjobs_refresh_time = 5; /* Seconds */
  ecl_job_status_type status = job_queue_null;

  if (__job != NULL) {
    lsf_job_type    * job    = (lsf_job_type    *) __job;
    lsf_driver_type * driver = (lsf_driver_type *) __driver;
    lsf_driver_assert_cast(driver); 
    lsf_job_assert_cast(job);
    

    /*
      The hash table driver->bjobs_output must be protected
    */
    pthread_mutex_lock( &driver->bjobs_lock ); 
    {
      if (difftime(time(NULL) , driver->last_bjobs_update) > bjobs_refresh_time) {
	lsf_driver_update_bjobs_table(driver);
	driver->last_bjobs_update = time( NULL );
      }
    
      if (hash_has_key( driver->bjobs_output , job->lsf_jobnr_char) ) 
	status = hash_get_int(driver->bjobs_output , job->lsf_jobnr_char);
      else
	/* 
	   It might be running - but sinjec job != NULL it is at least in the queue system.
	*/
	status = job_queue_pending;

    }
    pthread_mutex_unlock( &driver->bjobs_lock );
  }
  
  return status;
}

#endif



ecl_job_status_type lsf_driver_get_job_status(basic_queue_driver_type * __driver , basic_queue_job_type * __job) {
#ifdef LSF_LIBRARY_DRIVER
  return lsf_driver_get_job_status_libary(__driver , __job);
#else
  return lsf_driver_get_job_status_system(__driver , __job);
#endif
}


void lsf_driver_free_job(basic_queue_driver_type * __driver , basic_queue_job_type * __job) {
  lsf_job_type    * job    = (lsf_job_type    *) __job;
  lsf_driver_type * driver = (lsf_driver_type *) __driver;

  lsf_driver_assert_cast(driver); 
  lsf_job_assert_cast(job);
  lsf_job_free(job);
}



static void lsf_driver_killjob(int jobnr) {
#ifdef LSF_LIBRARY_DRIVER
  lsb_forcekilljob(jobnr);
#else
#endif
}


void lsf_driver_abort_job(basic_queue_driver_type * __driver , basic_queue_job_type * __job) {
  lsf_job_type    * job    = (lsf_job_type    *) __job;
  lsf_driver_type * driver = (lsf_driver_type *) __driver;
  lsf_driver_assert_cast(driver); 
  lsf_job_assert_cast(job);
  lsf_driver_killjob(job->lsf_jobnr);
  lsf_driver_free_job(__driver , __job);
}









basic_queue_job_type * lsf_driver_submit_job(basic_queue_driver_type * __driver, 
					     int   queue_index , 
					     const char * submit_cmd  	  , 
					     const char * run_path    	  , 
					     const char * job_name) {
  lsf_driver_type * driver = (lsf_driver_type *) __driver;
  lsf_driver_assert_cast(driver); 
  {
    lsf_job_type * job = lsf_job_alloc();
    char * lsf_stdout  = util_alloc_joined_string((const char *[2]) {run_path   , "/LSF.stdout"}  , 2 , "");
    char * command     = util_alloc_joined_string( (const char*[2]) {submit_cmd , run_path} , 2 , " "); 
    pthread_mutex_lock( &driver->submit_lock );
    
#ifdef LSF_LIBRARY_DRIVER
    driver->lsf_request.jobName = (char *) job_name;
    driver->lsf_request.outFile = lsf_stdout;
    driver->lsf_request.command = command;
    job->lsf_jobnr = lsb_submit( &driver->lsf_request , &driver->lsf_reply );
#else
    job->lsf_jobnr      = lsf_driver_submit_system_job( run_path , job_name , driver->queue_name , driver->resource_request , submit_cmd );
    job->lsf_jobnr_char = util_alloc_sprintf("%ld" , job->lsf_jobnr);
#endif

    pthread_mutex_unlock( &driver->submit_lock );
    free(lsf_stdout);
    free(command);

    
    if (job->lsf_jobnr > 0) {
      basic_queue_job_type * basic_job = (basic_queue_job_type *) job;
      basic_queue_job_init(basic_job);
      return basic_job;
    } else {
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



void * lsf_driver_alloc(const char * queue_name , int num_resource_request , const char ** resource_request_list) {
  lsf_driver_type * lsf_driver = util_malloc(sizeof * lsf_driver , __func__);
  lsf_driver->queue_name       = util_alloc_string_copy(queue_name);
  lsf_driver->__lsf_id         = LSF_DRIVER_ID;
  lsf_driver->submit           = lsf_driver_submit_job;
  lsf_driver->get_status       = lsf_driver_get_job_status;
  lsf_driver->abort_f          = lsf_driver_abort_job;
  lsf_driver->free_job         = lsf_driver_free_job;
  lsf_driver->free_driver      = lsf_driver_free__;
  lsf_driver->resource_request = util_alloc_joined_string(resource_request_list , num_resource_request , " ");
  pthread_mutex_init( &lsf_driver->submit_lock , NULL );
  
#ifdef LSF_LIBRARY_DRIVER
  memset(&lsf_driver->lsf_request , 0 , sizeof (lsf_driver->lsf_request));
  lsf_driver->lsf_request.options   	   = SUB_QUEUE + SUB_RES_REQ + SUB_JOB_NAME + SUB_OUT_FILE;
  lsf_driver->lsf_request.queue     	   = lsf_driver->queue_name;
  lsf_driver->lsf_request.resReq    	   = lsf_driver->resource_request;
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
    util_abort("%s failed to initialize LSF environment - aborting\n",__func__);
  setenv("BSUB_QUIET" , "yes" , 1);
#else
  pthread_mutex_init( &lsf_driver->bjobs_lock , NULL );
  lsf_driver->last_bjobs_update = time( NULL );
  lsf_driver->bjobs_output 	  = hash_alloc(); 
  lsf_driver->status_map   	  = hash_alloc();
  hash_insert_int(lsf_driver->status_map , "PEND"   , job_queue_pending);
  hash_insert_int(lsf_driver->status_map , "SSUSP"  , job_queue_running);
  hash_insert_int(lsf_driver->status_map , "PSUSP"  , job_queue_pending);
  hash_insert_int(lsf_driver->status_map , "RUN"    , job_queue_running);
  hash_insert_int(lsf_driver->status_map , "EXIT"   , job_queue_exit);
  hash_insert_int(lsf_driver->status_map , "USUSP"  , job_queue_running);
  hash_insert_int(lsf_driver->status_map , "DONE"   , job_queue_done);
  hash_insert_int(lsf_driver->status_map , "UNKWN"  , job_queue_exit); /* Uncertain about this one */
  {
    char * tmp_request = util_alloc_string_copy(lsf_driver->resource_request);
    lsf_driver->resource_request = util_realloc(lsf_driver->resource_request , strlen(tmp_request) + 5 , __func__);
    sprintf(lsf_driver->resource_request , "-R\"%s\"" , tmp_request);
    free(tmp_request);
  }
#endif
  {
    basic_queue_driver_type * basic_driver = (basic_queue_driver_type *) lsf_driver;
    basic_queue_driver_init(basic_driver);
    return basic_driver;
  }
}

void lsf_driver_free(lsf_driver_type * driver) {
  free(driver->resource_request);
  free(driver->queue_name);
  free(driver);
#ifdef LSF_SYSTEM_DRIVER
  hash_free(driver->status_map);
  hash_free(driver->bjobs_output);
#endif
  driver = NULL;
}

 void lsf_driver_free__(basic_queue_driver_type * driver) {
  lsf_driver_free((lsf_driver_type *) driver);
}

#undef LSF_DRIVER_ID  
#undef LSF_JOB_ID    

/*****************************************************************/

