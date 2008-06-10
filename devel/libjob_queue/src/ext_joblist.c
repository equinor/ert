#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <util.h>
#include <hash.h>
#include <ext_job.h>
#include <ext_joblist.h>


#define MODULE_NAME    "jobs.py"
#define JOBLIST_NAME   "jobList"


/*****************************************************************/

struct ext_joblist_struct {
  hash_type * jobs;
};



ext_joblist_type * ext_joblist_alloc() {
  ext_joblist_type * joblist = util_malloc( sizeof * joblist , __func__ );
  joblist->jobs = hash_alloc();
  return joblist;
}


void ext_joblist_free(ext_joblist_type * joblist) {
  hash_free(joblist->jobs);
  free(joblist);
}


ext_job_type * ext_joblist_alloc_new(ext_joblist_type * joblist , const char * new_name) {
  ext_job_type * new_job = ext_job_alloc( new_name );
  hash_insert_hash_owned_ref(joblist->jobs , new_name , new_job , ext_job_free__);
  return new_job;
}


ext_job_type * ext_joblist_add_job(ext_joblist_type * joblist , const char * name , const char * config_file) {
  ext_job_type * new_job = ext_job_fscanf_alloc(name , config_file);
  hash_insert_hash_owned_ref(joblist->jobs , name , new_job , ext_job_free__);
  return new_job;
}


ext_job_type * ext_joblist_get_job(const ext_joblist_type * joblist , const char * job_name) {
  if (hash_has_key(joblist->jobs , job_name))
    return hash_get(joblist->jobs , job_name);
  else {
    util_abort("%s: asked for job:%s which does not exist\n",__func__ , job_name);
    return NULL;
  }
}

bool ext_joblist_has_job(const ext_joblist_type * joblist , const char * job_name) {
  return hash_has_key(joblist->jobs , job_name);
}


void ext_joblist_python_fprintf(const ext_joblist_type * joblist , const char ** kw_list , int size , const char * path, const hash_type * context) {
  char * module_file = util_alloc_full_path(path , MODULE_NAME);
  FILE * stream      = util_fopen(module_file , "w");
  int i;

  fprintf(stream , "%s = [" , JOBLIST_NAME);
  for (i=0; i < size; i++) {
    const ext_job_type * job = ext_joblist_get_job( joblist , kw_list[i] );
    ext_job_python_fprintf(job , stream , context);
    if (i < (size - 1))
      fprintf(stream,",\n");
  }
  fprintf(stream , "]\n");
  
  fclose(stream);
  free(module_file);
}
