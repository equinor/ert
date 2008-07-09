#include <stdlib.h>
#include <stdbool.h>
#include <util.h>
#include <string.h>
#include <stdio.h>
#include <enkf_sched.h>
#include <sched_file.h>
#include <stringlist.h>
#include <ext_joblist.h>




struct enkf_sched_node_struct {
  int    	    report_step1;
  int    	    report_step2;
  int               report_stride; 
  bool              enkf_active;
  stringlist_type * forward_model;
};





struct enkf_sched_struct {
  stringlist_type       * std_forward_model;
  int    size;
  enkf_sched_node_type ** node_list;
};


/** 
    allocating a fully invalid node 
*/
static enkf_sched_node_type * enkf_sched_node_alloc_empty() {
  enkf_sched_node_type * sched_node = util_malloc(sizeof * sched_node , __func__);
  sched_node->report_step1  = -1;
  sched_node->report_step2  = -1;
  sched_node->report_stride = 0;
  sched_node->enkf_active   = false;
  sched_node->forward_model = NULL;
  return sched_node;
}


/*****************************************************************/

/**
   This function scans a stream for the info it needs to allocate a
   enkf_sched_node_type instance. The format expected on the stream is as follows:

   REPORT_STEP1   REPORT_STEP2   ON|OFF   <STRIDE>   <FORWARD_MODEL>

   Observe the following:

   * If the list contains four or more items, we try to interpret item
     number four as a stride. If that succeeds we use it as stride,
     otherwise it is assumed to be part of the forward model. (i.e if
     the forward model starts with an element which can be interpreted
     as an integer - i.e. only containing digits, you *MUST* enter a
     value for the stride.)

   * If no stride is found, a default stride is used. If the value of
     enkf_update is on, the default stride equals 1, otherwise it
     equals report_step2 - report_step1.

   * If no forward_model is found, the default forward model is used.

   * If the stream is positioned at an empty line NULL is returned. No
     comments (at present).
*/


static enkf_sched_node_type * enkf_sched_node_fscanf_alloc(FILE * stream, stringlist_type * default_forward_model, const ext_joblist_type * joblist) {
  enkf_sched_node_type * sched_node;
  bool at_eof;
  char ** token_list;
  bool enkf_active;
  int report_step1 , report_step2, report_stride;
  int tokens;
  stringlist_type * forward_model = default_forward_model;
  char * line = util_fscanf_alloc_line(stream , &at_eof);
  if (line != NULL) {
    util_split_string(line , " \t" , &tokens , &token_list);
    if (tokens < 3) 
      util_abort("%s: fatal error when parsing line:\'%s\' - must have at least 3 tokens \n",__func__ , line);
    if (util_sscanf_int(token_list[0] , &report_step1) && util_sscanf_int(token_list[1] , &report_step2)) {
      util_strupr(token_list[2]);
      if (strcmp(token_list[2] , "ON") == 0) {
	enkf_active = true;
	report_stride = 1;
      } else if (strcmp(token_list[2] , "OFF") == 0) {
	enkf_active = false;
	report_stride = report_step2 - report_step1;
      } else 
	util_abort("%s: failed to interpret %s as ON || OFF \n",__func__ , token_list[2]);

      if (tokens > 3) {
	int model_start;
	int model_length;
	if (util_sscanf_int(token_list[3] , &report_stride)) 
	  model_start  = 4;
	else
	  model_start = 3;
	model_length = tokens - model_start + 1;
	if (model_length > 0)
	  forward_model = stringlist_alloc_argv_copy((const char **) &token_list[model_start] , model_length);
      }
    } else
      util_abort("%s: failed to parse %s and %s as integers\n",__func__ , token_list[0] , token_list[1]);

    sched_node = enkf_sched_node_alloc_empty();
    sched_node->report_step1  = report_step1;
    sched_node->report_step2  = report_step2;
    sched_node->enkf_active   = enkf_active;
    sched_node->report_stride = report_stride;
    sched_node->forward_model = forward_model;

    if (forward_model != default_forward_model) {
      int argc = stringlist_get_size(forward_model);
      int i;
      for (i = 0; i < argc; i++)
	if (!ext_joblist_has_job(joblist , stringlist_iget(forward_model , i)))
	  util_abort("%s: the forward job:%s has not been installed\n",__func__ , stringlist_iget(forward_model , i));
    }
  }
}




/*****************************************************************/





