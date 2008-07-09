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
  const stringlist_type * forward_model;
};





struct enkf_sched_struct {
  int    size;                                /* NUmber of elements in the node_lost */
  enkf_sched_node_type ** node_list;

  const stringlist_type  * std_forward_model;
  int                      schedule_num_reports; /* The number of DATES / TSTEP keywords in the currently active SCHEDULE file. */
  int                      last_report;          /* The last report_step in the enkf_sched instance - can be greater than schedule_num_reports .*/
  const ext_joblist_type * joblist; 
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

static enkf_sched_node_type * enkf_sched_node_alloc(int report_step1 , int report_step2 , int report_stride , bool enkf_active , const stringlist_type * forward_model) {
  enkf_sched_node_type * node = enkf_sched_node_alloc_empty();
  node->report_step1  = report_step1;
  node->report_step2  = report_step2;
  node->report_stride = report_stride;
  node->enkf_active   = enkf_active;
  node->forward_model = forward_model;
  return node;
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
	  util_abort("%s: the forward job:%s has not been installed.\n",__func__ , stringlist_iget(forward_model , i));
    }
  }
}

/*****************************************************************/

void enkf_sched_idel_node(enkf_sched_type * enkf_sched , int index) {
  if (index < 0 || index >= enkf_sched->size)
    util_abort("%s: index:%d invlid. Valid range: [0,%d) \n",__func__ , index ,  enkf_sched->size);

  {
    int new_size = enkf_sched->size - 1;
    enkf_sched_node_type ** new_list = util_malloc(new_size * sizeof * new_list , __func__);
    memcpy(new_list          ,  enkf_sched->node_list          , index * sizeof * new_list);  /* Copying everything _before_ index */
    memcpy(&new_list[index]  , &enkf_sched->node_list[index+1] , (new_size - index) * sizeof * new_list);  /* Copying everything _after_ index */
    free( enkf_sched->node_list );
    enkf_sched->node_list = new_list;
    enkf_sched->size      = new_size;
  }
}

/**
   When we are adding a new node with two values of report_step1 and
   report_step2 there are many different possibilities:

   1. report_step > last_report: In this case we abort, because there
      will be a report interval [last_report , report_step> where we
      do not know what to do.

   2. report_step1 agrees with the report_step1 on an existing node:
 
      a. The new step is fully contained within an existing node, with
         new(report_step2) < existing(report_step2):

         Existing: -----<1>----------------------------<2>-------
         New:           <1>--------<2>
      
         What we must do here is the following:
   
         * We must shift report_step1 of the existing node to match
           report_step2 of the new node.
 
	 * The new node must be inserted in the list before the
	   existing node.

      b. The new node agrees fully with an existing node:

         Existing: -----<1>----------------------------<2>-------
         New:           <1>----------------------------<2>

         In this case we must we must _replace_ the existing node with
         the new one.
      
      c. The new node extends well into nodes forward in time:

                             A             B                C
         Existing: -----<1>-------<2|1>----------<2|1>----------
         New:           <1>---------------------------------<2>     

         * We must remove existing nodes A and B.

	 * We must adjust report_step1 of node C to match report_step2
	   of the new node.

	 * Insert new node before existing node C.

   3. report_step1 does not agree with an existing node:

      a. The new report_step is fully contained within an existing
         node:

         Existing: -----<1>----------------------------<2>-------
         New:                    <1>--------<2>
       
         * We must adjust report_step2 of the existing step to match
           the new report_step1.
            
	 * We append the new item after the existing item.
   
         * We add new item from the new report_step2 -> old report_step2.  

      b. The new report_step2 agrees exactly with an existing report2.

  
*/


static void enkf_sched_add_node(enkf_sched_type * enkf_sched , enkf_sched_node_type * node) {
  if (enkf_sched->size == 0) {
    enkf_sched->size = 1;
    enkf_sched->node_list = util_malloc(sizeof * enkf_sched->node_list , __func__);
    enkf_sched->node_list[0] = node;
  } else {
    int i;
    int new_report1 = node->report_step1;
    int new_report2 = node->report_step2;
    util_exit("%s: sorry not implemented yet ... \n");
    


  }
}

static enkf_sched_type * enkf_sched_alloc_empty( const sched_file_type * sched_file , const ext_joblist_type * joblist , const stringlist_type * forward_model) {
  enkf_sched_type * enkf_sched = util_malloc(sizeof * enkf_sched , __func__);
  enkf_sched->node_list 	   = NULL;
  enkf_sched->size      	   = 0;       
  enkf_sched->std_forward_model    = forward_model;
  enkf_sched->joblist              = joblist;
  enkf_sched->schedule_num_reports = sched_file_count_report_steps( sched_file );
  enkf_sched->last_report          = 0;
  return enkf_sched;
  
}



static void  enkf_sched_set_default(enkf_sched_type * enkf_sched ) {
  enkf_sched_node_type * node = enkf_sched_node_alloc(0 , enkf_sched->schedule_num_reports , 1 , true , enkf_sched->std_forward_model);
  enkf_sched_add_node(enkf_sched , node);
}

/*****************************************************************/

/**
   This functions parses a config file, and builds a enkf_sched_type *
   instance from it. If the filename argument is NULL a default
   enkf_sched_type instance is allocated.
*/

enkf_sched_type * enkf_sched_fscanf_alloc(const char * enkf_sched_file , const sched_file_type * sched_file , const ext_joblist_type * joblist, const stringlist_type * default_forward_model) {
  
  enkf_sched_type * enkf_sched = enkf_sched_alloc_empty(sched_file , joblist ,default_forward_model);
  enkf_sched_set_default(enkf_sched);
  if (enkf_sched_file != NULL) {
  }
  
  return enkf_sched;
}



