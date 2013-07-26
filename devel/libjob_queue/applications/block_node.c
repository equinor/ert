/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'block_node.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <lsf/lsbatch.h>

#include <ert/util/stringlist.h>
#include <ert/util/util.h>
#include <ert/util/hash.h>
#include <ert/util/vector.h>

#include <ert/job_queue/lsf_driver.h>

#define BLOCK_COMMAND        "/d/proj/bg/enkf/bin/block_node.py"
#define STATOIL_LSF_REQUEST  "select[cs && x86_64Linux]"

typedef struct {
  lsf_job_type    * lsf_job;
  stringlist_type * hostlist;
  bool              running;
  bool              block_job;
} block_job_type;


block_job_type * block_job_alloc() {
  block_job_type * job = util_malloc( sizeof * job );

  job->lsf_job   = NULL;
  job->running   = false;
  job->block_job = false;
  job->hostlist  = stringlist_alloc_new();

  return job;
}


void block_job_free( block_job_type * block_job ) {
  stringlist_free( block_job->hostlist );
  if (block_job->lsf_job)
    lsf_job_free( block_job->lsf_job );
  
  free( block_job );
}


  
/*typedef struct {
  struct  submit         lsf_request;
  struct  submitReply    lsf_reply; 
  long    int            lsf_jobnr; 
  char                 * host_name;
  const   char         * command;
  bool                   running;
  bool                   block_job; 
} lsf_job_type;
*/


/*
lsf_job_type * lsf_job_alloc(char * queue_name) { 
  lsf_job_type * lsf_job = util_malloc( sizeof * lsf_job );
  
  memset(&lsf_job->lsf_request , 0 , sizeof (lsf_job->lsf_request));
  lsf_job->lsf_request.queue            = queue_name;
  lsf_job->lsf_request.beginTime        = 0;
  lsf_job->lsf_request.termTime         = 0;   
  lsf_job->lsf_request.numProcessors    = 1;
  lsf_job->lsf_request.maxNumProcessors = 1;
  lsf_job->lsf_request.command         = BLOCK_COMMAND; 
  lsf_job->lsf_request.outFile         = "/tmp/lsf";
  lsf_job->lsf_request.jobName         = "lsf";
  lsf_job->lsf_request.resReq          = STATOIL_LSF_REQUEST;
  {
    int i;
    for (i=0; i < LSF_RLIM_NLIMITS; i++) 
      lsf_job->lsf_request.rLimits[i] = DEFAULT_RLIMIT;
  }
  lsf_job->lsf_request.options2 = 0;
  lsf_job->lsf_request.options  = SUB_QUEUE + SUB_JOB_NAME + SUB_OUT_FILE + SUB_RES_REQ;
  lsf_job->host_name            = NULL;
  lsf_job->running              = false;
  lsf_job->block_job            = false;
  return lsf_job;
}
*/


void update_job_status( lsf_driver_type * driver , block_job_type * job , hash_type * nodes) {
  if (!job->running) {
    int lsf_status = lsf_driver_get_job_status_lsf( driver , job->lsf_job );
    if (lsf_status == JOB_STAT_RUN) {
      lsf_job_export_hostnames( job->lsf_job , job->hostlist ); 
      {
        int ihost;
        printf("%ld running on: ",lsf_job_get_jobnr( job->lsf_job) );
        for (ihost = 0; ihost < stringlist_get_size( job->hostlist ); ihost++) {
          const char * host = stringlist_iget( job->hostlist , ihost );
          printf("%s ",host);
          if (hash_has_key( nodes, host))  { /* This is one of the instances which should be left running. */
            job->block_job = true;
            printf("Got a block on:%s \n" , host );
          }
        }
        printf("\n");
      }
      job->running = true;
    }
  }
}



/*****************************************************************/



void add_jobs(lsf_driver_type * driver , vector_type * job_pool , int num_cpu , int chunk_size) {
  int i;
  char * cwd = util_alloc_cwd();
  for (i=0; i < chunk_size; i++) {
    block_job_type * job = block_job_alloc();
    job->lsf_job = lsf_driver_submit_job(driver , BLOCK_COMMAND , num_cpu , cwd , "BLOCK" , 0 , NULL );
    vector_append_ref( job_pool , job );
  }
  free( cwd );
}


void update_pool_status(lsf_driver_type * driver , vector_type * job_pool , hash_type * block_nodes , int * blocked , int * pending) {
  int i;
  int block_count = 0;
  int pend_count  = 0;
  for (i=0; i < vector_get_size( job_pool ); i++) {
    block_job_type * job = vector_iget( job_pool , i );
    update_job_status( driver , job , block_nodes );
    if (!job->running)
      pend_count++;
    else {
      if (job->block_job)
        block_count++;
    }
  }
  *blocked = block_count;
  *pending = pend_count;
}


void kill_jobs( lsf_driver_type * driver , const vector_type * job_pool) {
  int job_nr;
  for (job_nr = 0; job_nr < vector_get_size( job_pool ); job_nr++) {
    block_job_type * job = vector_iget( job_pool , job_nr );
    
    if (job->block_job) {
      printf("Job:%ld is running on hosts: ", lsf_job_get_jobnr( job->lsf_job ));
      stringlist_fprintf( job->hostlist , " " , stdout );
      printf("\n");
    } else
      lsf_driver_kill_job( driver , job->lsf_job );
    
    block_job_free( job );
  }
}


int main( int argc, char ** argv) {
  if (argc == 1)
    util_exit("block_node  node1  node2  node3:2  \n");
  
  /* Initialize lsf environment */
  util_setenv( "LSF_BINDIR"    , "/prog/LSF/8.0/linux2.6-glibc2.3-x86_64/bin" );
  util_setenv( "LSF_LINDIR"    , "/prog/LSF/8.0/linux2.6-glibc2.3-x86_64/lib" );
  util_setenv( "XLSF_UIDDIR"   , "/prog/LSF/8.0/linux2.6-glibc2.3-x86_64/lib/uid" );
  util_setenv( "LSF_SERVERDIR" , "/prog/LSF/8.0/linux2.6-glibc2.3-x86_64/etc");
  util_setenv( "LSF_ENVDIR"    , "/prog/LSF/conf");
  
  util_update_path_var( "PATH"               , "/prog/LSF/8.0/linux2.6-glibc2.3-x86_64/bin" , false);
  util_update_path_var( "LD_LIBRARY_PATH"    , "/prog/LSF/8.0/linux2.6-glibc2.3-x86_64/lib" , false);

  
  lsf_driver_type * lsf_driver = lsf_driver_alloc();
  if (lsf_driver_get_submit_method( lsf_driver ) != LSF_SUBMIT_INTERNAL)
    util_exit("Sorry - the block_node program must be invoked on a proper LSF node \n");

  {
    hash_type * nodes       = hash_alloc();
    int         node_count  = 0; 
    int iarg;
    printf("Attempting to block nodes \n");
    for (iarg = 1; iarg < argc; iarg++) {
      char   node_name[64];
      int    num_slots;
      if (sscanf(argv[iarg] , "%s:%d" , node_name , &num_slots) != 2) 
        num_slots = 1;

      hash_insert_int( nodes , node_name , num_slots );
      node_count += num_slots;
      
      printf("  %s",node_name);
      if (num_slots != 1)
        printf(" * %d",num_slots );
      printf("\n");
    }
    printf("-----------------------------------------------------------------\n");
    
    {
      const int sleep_time    = 5;
      const int max_attempt   = 100;
      const int chunk_size    = 25;   /* We submit this many at a time. */
      const int max_pool_size = 1000; /* The absolute total maximum of jobs we will submit. */  
      const int num_cpu       = 4;

      vector_type  * job_pool    = vector_alloc_new();
      bool           cont        = true;
      int            pending     = 0;   
      int            blocked;
      int            attempt     = 0;
      while (cont) {
        printf("Attempt: %2d/%2d ",attempt , max_attempt); fflush( stdout );
        if (pending == 0) {
          if (vector_get_size( job_pool ) < max_pool_size)
            add_jobs( lsf_driver , job_pool , num_cpu , chunk_size );
        }
        
        update_pool_status( lsf_driver , job_pool , nodes , &blocked , &pending);
        if (blocked == node_count)
          cont = false;                                         /* Ok - we have got them all blocked - leave the building. */

        attempt++;
        if (attempt > max_attempt)
          cont = false;
        
        if (cont) sleep( sleep_time );
        printf("\n");
      }
      if (blocked < node_count)
        printf("Sorry - failed to block all the nodes \n");
      
      kill_jobs( lsf_driver , job_pool );
      hash_free( nodes );
    }
  }
}
