#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <util.h>
#include <hash.h>
#include <ext_job.h>
#include <stringlist.h>
#include <ext_joblist.h>
#include <subst.h>


#define MODULE_NAME    "jobs.py"
#define JOBLIST_NAME   "jobList"

/**
   About the 'license' system:
   ---------------------------

   There is a simple possibility to limit the number of jobs which are
   running in parallell. It works like this:

    1. For the joblist as a whole a license_path is created. This
       license path should contain both a uid and pid of the current
       process. This ensures that:

       a. The license count is per user and per ert instance.
       b. Each ert instance starts with a fresh license count. A
          license path, and license files left dangling after unclean
          shutdown can just be removed.

    2. For each job in the joblist a subdirectory is created under the
       license_path. 

    3. For each job a license_file is created, and for each time a new
       instance is checked out a hard_link to this license_file is
       created - i.e. the number of checked out licenses is a
       hard_link count (-1). 
       
       Step three here is implemented by the job_dispatch script
       actually running the jobs.

*/



/*****************************************************************/

struct ext_joblist_struct {
  hash_type * jobs;
  char      * license_root_path;   
};




/**
   It is essential that the license_root_path is on a volume which is
   accessible from all the nodes which will run jobs. Using e.g. /tmp
   as license_root_path will fail HARD.
*/


ext_joblist_type * ext_joblist_alloc(const char * license_root_path) {
  ext_joblist_type * joblist = util_malloc( sizeof * joblist , __func__ );
  const char       * user    = getenv("USER"); 
  joblist->jobs = hash_alloc();
  /**
    Appending /user/pid to the license root path. Everything
    including the pid is removed when exiting (gracefully ...).
    
    Dangling license directories after a crash can just be removed.
  */
  joblist->license_root_path = util_alloc_sprintf("%s%c%s%c%d" , license_root_path , UTIL_PATH_SEP_CHAR , user , UTIL_PATH_SEP_CHAR , getpid());
  return joblist; 
}


void ext_joblist_free(ext_joblist_type * joblist) {
  hash_free(joblist->jobs);
  util_unlink_path( joblist->license_root_path );
  util_safe_free( joblist->license_root_path );
  free(joblist);
}



ext_job_type * ext_joblist_add_job(ext_joblist_type * joblist , const char * name , const char * config_file) {
  ext_job_type * new_job = ext_job_fscanf_alloc(name , joblist->license_root_path , config_file); /* Return NULL if you did not have permission to read file. */
  if (new_job != NULL) 
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


ext_job_type * ext_joblist_get_job_copy(const ext_joblist_type * joblist , const char * job_name) {
  if (hash_has_key(joblist->jobs , job_name)) 
    return ext_job_alloc_copy(hash_get(joblist->jobs , job_name));
  else {
    util_abort("%s: asked for job:%s which does not exist\n",__func__ , job_name);
    return NULL;
  }
}


bool ext_joblist_has_job(const ext_joblist_type * joblist , const char * job_name) {
  return hash_has_key(joblist->jobs , job_name);
}


void ext_joblist_python_fprintf(const ext_joblist_type * joblist , const stringlist_type * kw_list , const char * path, const subst_list_type * subst_list) {
  char * module_file = util_alloc_filename(path , MODULE_NAME , NULL);
  FILE * stream      = util_fopen(module_file , "w");
  int i;

  fprintf(stream , "%s = [" , JOBLIST_NAME);
  for (i=0; i < stringlist_get_size(kw_list); i++) {
    const ext_job_type * job = ext_joblist_get_job( joblist , stringlist_iget(kw_list , i));
    ext_job_python_fprintf(job , stream , subst_list);
    if (i < (stringlist_get_size(kw_list) - 1))
      fprintf(stream,",\n");
  }
  fprintf(stream , "]\n");
  
  fclose(stream);
  free(module_file);
}
