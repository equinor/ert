#include <lsf/lsbatch.h>
#include <stdlib.h>
#include <util.h>
#include <string.h>
#include <lsf_request.h>
#include <hash.h>
#include <vector.h>

#define BLOCK_COMMAND "/d/proj/bg/enkf/bin/block_node.py"




typedef struct {
  struct  submit         lsf_request;
  struct  submitReply    lsf_reply; 
  long    int            lsf_jobnr; 
  char                 * host_name;
  const   char         * command;
  bool                   running;
  bool                   block_job; 
} lsf_job_type;




lsf_job_type * lsf_job_alloc(char * queue_name) { 
  lsf_job_type * lsf_job = util_malloc( sizeof * lsf_job , __func__);
  
  memset(&lsf_job->lsf_request , 0 , sizeof (lsf_job->lsf_request));
  lsf_job->lsf_request.queue     	= queue_name;
  lsf_job->lsf_request.beginTime 	= 0;
  lsf_job->lsf_request.termTime   	= 0;   
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



void update_job_status( lsf_job_type * job , hash_type * nodes) {
  if (!job->running) {
    struct jobInfoEnt *job_info;
    if (lsb_openjobinfo(job->lsf_jobnr , NULL , NULL , NULL , NULL , ALL_JOB) == 1) {
      job_info = lsb_readjobinfo( NULL );
      lsb_closejobinfo();
      
      if (job_info->status == JOB_STAT_RUN) {
        job->running   = true;
        job->host_name = util_realloc_string_copy( job->host_name , job_info->exHosts[0]); /* Hardcoded only one node. */

        if (hash_has_key( nodes, job->host_name))  {/* This is one of the instances which should be left running. */
          job->block_job = true;
          printf("Got a block on:%s \n",job->host_name);
        }
      }
    }
  }
}



void lsf_job_submit( lsf_job_type * job ) {
  job->lsf_jobnr = lsb_submit( &job->lsf_request , &job->lsf_reply );
}




void lsf_job_kill( lsf_job_type * job ) {
  lsb_forcekilljob(job->lsf_jobnr);
}



void lsf_job_free( lsf_job_type * job ) {
  if (job->block_job)
    printf("Node: %-16s blocked by job:%ld \n",job->host_name , job->lsf_jobnr);
  else
    lsf_job_kill( job );
  
  util_safe_free( job->host_name );
  free( job );
}


void lsf_job_free__( void * arg) {
  lsf_job_free( (lsf_job_type * ) arg);
}

/*****************************************************************/



void add_jobs(vector_type * job_pool , int chunk_size) {
  int i;
  for (i=0; i < chunk_size; i++) {
    lsf_job_type * job = lsf_job_alloc("normal");
    vector_append_owned_ref( job_pool , job , lsf_job_free__);
    lsf_job_submit( job );
  }
}


void update_pool_status(vector_type * job_pool , hash_type * block_nodes , int * blocked , int * pending) {
  int i;
  int block_count = 0;
  int pend_count  = 0;
  for (i=0; i < vector_get_size( job_pool ); i++) {
    lsf_job_type * job = vector_iget( job_pool , i );
    update_job_status( job , block_nodes );
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




int main( int argc, char ** argv) {
  if (argc == 1)
    util_exit("block_node  node1  node2  node3:2  \n");
  
  /* Initialize lsf environment */
  setenv( "LSF_BINDIR"    , "/prog/LSF/7.0/linux2.6-glibc2.3-x86_64/bin" , 1);
  setenv( "LSF_LINDIR"    , "/prog/LSF/7.0/linux2.6-glibc2.3-x86_64/lib" , 1);
  setenv( "XLSF_UIDDIR"   , "/prog/LSF/7.0/linux2.6-glibc2.3-x86_64/lib/uid" , 1);
  setenv( "LSF_SERVERDIR" , "/prog/LSF/7.0/linux2.6-glibc2.3-x86_64/etc" , 1);
  setenv( "LSF_ENVDIR"    , "/prog/LSF/conf" , 1);
  
  util_update_path_var( "PATH"               , "/prog/LSF/7.0/linux2.6-glibc2.3-x86_64/bin" , false);
  util_update_path_var( "LD_LIBRARY_PATH"    , "/prog/LSF/7.0/linux2.6-glibc2.3-x86_64/lib" , false);

  if (lsb_init(NULL) != 0) 
    util_abort("%s failed to initialize LSF environment : %s/%d  \n",__func__ , lsb_sysmsg() , lsberrno);
  setenv("BSUB_QUIET" , "yes" , 1);
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
      const int max_attempt   = 25;
      const int chunk_size    = 10;  /* We submit this many at a time. */
      const int max_pool_size = 100; /* The absolute total maximum of jobs we will submit. */  

      vector_type  * job_pool    = vector_alloc_new();
      bool           cont        = true;
      int            pending     = 0;   
      int            blocked;
      int            attempt     = 0;
      while (cont) {
        printf("Attempt: %2d/%2d ",attempt , max_attempt); fflush( stdout );
        if (pending == 0) {
          if (vector_get_size( job_pool ) < max_pool_size)
            add_jobs( job_pool , chunk_size );
        }
        
        update_pool_status( job_pool , nodes , &blocked , &pending);
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
      
      vector_free( job_pool );
      hash_free( nodes );
    }
  }
}
