#include <stdlib.h>
#include <stdbool.h>
#include <util.h>
#include <string.h>
#include <stdio.h>
#include <enkf_types.h>
#include <enkf_sched.h>
#include <stringlist.h>
#include <ext_joblist.h>
#include <forward_model.h>
#include <vector.h>


struct enkf_sched_node_struct {
  int    	       report_step1;
  int    	       report_step2;
  bool                 enkf_active;
  forward_model_type * forward_model;   /* Will mostly be NULL */
};





struct enkf_sched_struct {
  vector_type    *nodes;                 /* Vector consisting of enkf_sched_node_type instances. */
  int             schedule_num_reports;  /* The number of DATES / TSTEP keywords in the currently active SCHEDULE file. */
  int             last_report;           /* The last report_step in the enkf_sched instance - can be greater than schedule_num_reports .*/
};




static enkf_sched_node_type * enkf_sched_node_alloc(int report_step1 , int report_step2 , bool enkf_active , forward_model_type * forward_model) {
  enkf_sched_node_type * node = util_malloc(sizeof * node , __func__);
  node->report_step1  = report_step1;
  node->report_step2  = report_step2;
  node->enkf_active   = enkf_active;
  if (forward_model != NULL) 
    util_abort("%s : Sorry - support for special forward models has been (temporarily) removed\n",__func__);
  
  node->forward_model = forward_model; 
  return node;
}



void enkf_sched_node_get_data(const enkf_sched_node_type * node , int * report_step1 , int * report_step2 , bool * enkf_on , forward_model_type ** forward_model) {
  *report_step1    = node->report_step1;
  *report_step2    = node->report_step2;
  *enkf_on         = node->enkf_active;
  *forward_model   = node->forward_model;
}

int enkf_sched_node_get_last_step(const enkf_sched_node_type * node) {
  return node->report_step2;
}



static void enkf_sched_node_free(enkf_sched_node_type * node) {
  if (node->forward_model != NULL)
    forward_model_free( node->forward_model);

  free(node);
}

static void enkf_sched_node_free__(void * arg) {
  enkf_sched_node_free( (enkf_sched_node_type *) arg );
}




static void enkf_sched_node_fprintf(const enkf_sched_node_type * node , FILE * stream) {
  if (node->enkf_active)
    fprintf(stream , "%4d   %4d   %s     ",node->report_step1 , node->report_step2 , "ON ");
  else
    fprintf(stream , "%4d   %4d   %s     ",node->report_step1 , node->report_step2 , "OFF");
  
  if (node->forward_model != NULL)
    forward_model_fprintf( node->forward_model  , stream );
  else
    fprintf(stream, "*");
  
  fprintf(stream , "\n");
}




/*****************************************************************/

static void enkf_sched_append_node(enkf_sched_type * enkf_sched , enkf_sched_node_type * new_node) {
  vector_append_owned_ref(enkf_sched->nodes , new_node , enkf_sched_node_free__);
  enkf_sched->last_report = new_node->report_step2;
}


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

   * If no stride is found, a default stride is used - the default
     stride is report_step2 - report_step1 (i.e. the whole thing in
     one go.)

   * If no forward_model is found, the default forward model is used.

   * If the stream is positioned at an empty line NULL is returned. No
     comments are supported.
*/


static void  enkf_sched_fscanf_alloc_nodes(enkf_sched_type * enkf_sched , FILE * stream , const ext_joblist_type * joblist , bool use_lsf , bool * at_eof) {
  forward_model_type * forward_model = NULL;
  enkf_sched_node_type * sched_node  = NULL;
  char ** token_list;
  bool enkf_active = false; /* Compiler shut up */
  int report_step1 , report_step2, report_stride;
  int tokens;
  
  char * line = util_fscanf_alloc_line(stream , at_eof);
  if (line != NULL) {
    util_split_string(line , " \t" , &tokens , &token_list);
    if (tokens >= 3) {
      if (util_sscanf_int(token_list[0] , &report_step1) && util_sscanf_int(token_list[1] , &report_step2)) {
	util_strupr(token_list[2]);
		  
	report_stride = report_step2 - report_step1;
	if (strcmp(token_list[2] , "ON") == 0) 
	  enkf_active = true;
	else if (strcmp(token_list[2] , "OFF") == 0) 
	  enkf_active = false;
	else 
	  util_abort("%s: failed to interpret %s as ON || OFF \n",__func__ , token_list[2]);
	
	if (tokens > 3) {
	  int model_start;
	  int model_length;
	  if (util_sscanf_int(token_list[3] , &report_stride)) 
	    model_start  = 4;
	  else
	    model_start = 3;
	  model_length = tokens - model_start;
	  if (model_length > 0) {
	    char * input_string = util_alloc_joined_string( (const char **) &token_list[model_start] , model_length , " ");
	    forward_model = forward_model_alloc( input_string , joblist , use_lsf );
	    free( input_string );
	  }
	}
      } else
	util_abort("%s: failed to parse %s and %s as integers\n",__func__ , token_list[0] , token_list[1]);
      {
	/* Adding node(s): */
	int step1 = report_step1;
	int step2;
	
	do {
	  step2 = util_int_min(step1 + report_stride , report_step2);
	  sched_node = enkf_sched_node_alloc(step1 , step2 , enkf_active , forward_model);
	  step1 = step2;
	  enkf_sched_append_node( enkf_sched , sched_node);
	} while (step2 < report_step2);
      }
    }
    util_free_stringlist(token_list , tokens);
    free(line);
  } 
}


/*****************************************************************/



static void enkf_sched_verify__(const enkf_sched_type * enkf_sched) {
  int index;
  const enkf_sched_node_type * first_node = vector_iget_const( enkf_sched->nodes , 0);
  if (first_node->report_step1 != 0)
    util_abort("%s: must start at report-step 0 \n",__func__);
  
  for (index = 0; index < (vector_get_size(enkf_sched->nodes) - 1); index++) {
    const enkf_sched_node_type * node1 = vector_iget_const( enkf_sched->nodes , index );
    const enkf_sched_node_type * node2 = vector_iget_const( enkf_sched->nodes , index + 1);
    int report1      = node1->report_step1;
    int report2      = node1->report_step2;
    int next_report1 = node2->report_step1;
    
    if (report1 >= report2) {
      enkf_sched_fprintf(enkf_sched , stdout);
      util_abort("%s: enkf_sched step of zero/negative length:%d - %d - that is not allowed \n",__func__ , report1 , report2);
    }

    if (report2 != next_report1) {
      enkf_sched_fprintf(enkf_sched , stdout);
      util_abort("%s - abort \n",__func__);
    }
  }

  /* Verify that report_step2 > report_step1 also for the last node. */
  {
    index = vector_get_size(enkf_sched->nodes) - 1;
    const enkf_sched_node_type * node1 = vector_iget_const( enkf_sched->nodes , index );
    int report1      = node1->report_step1;
    int report2      = node1->report_step2;
    
    if (report2 <= report1)
      util_abort("%s: enkf_sched step of zero/negative length:%d - %d - that is not allowed \n",__func__ , report1 , report2);
      
  }
}





void enkf_sched_free( enkf_sched_type * enkf_sched) {
  vector_free( enkf_sched->nodes );
  free( enkf_sched );
}





static enkf_sched_type * enkf_sched_alloc_empty( int num_restart_files) {
  enkf_sched_type * enkf_sched     = util_malloc(sizeof * enkf_sched , __func__);
  enkf_sched->nodes                = vector_alloc_new();  
  enkf_sched->schedule_num_reports = num_restart_files - 1;
  enkf_sched->last_report          = 0;
  return enkf_sched;
}



static void  enkf_sched_set_default(enkf_sched_type * enkf_sched , run_mode_type run_mode) {
  enkf_sched_node_type * node;

  if (run_mode == enkf_assimilation) {
    /* Default enkf: stride one - active at all report steps. */
    /* Have to explicitly add all these nodes. */
    int report_step;
    for (report_step = 0; report_step < enkf_sched->schedule_num_reports; report_step++) {
      node = enkf_sched_node_alloc(report_step , report_step + 1, true , NULL);
      enkf_sched_append_node(enkf_sched , node);
    }
  } else {
    /* experiment: Do the whole thing in ONE go - off at all report steps. */
    node = enkf_sched_node_alloc(0 , enkf_sched->schedule_num_reports , false , NULL); 
    enkf_sched_append_node(enkf_sched , node);
  }
}



/**
   This functions parses a config file, and builds a enkf_sched_type *
   instance from it. If the filename argument is NULL a default
   enkf_sched_type instance is allocated.
*/

enkf_sched_type * enkf_sched_fscanf_alloc(const char * enkf_sched_file , int num_restart_files , int * last_report_step , run_mode_type run_mode, const ext_joblist_type * joblist , bool use_lsf) {
  enkf_sched_type * enkf_sched = enkf_sched_alloc_empty(num_restart_files);
  if (enkf_sched_file == NULL)
    enkf_sched_set_default(enkf_sched , run_mode);
  else {
    FILE * stream = util_fopen(enkf_sched_file , "r");
    bool at_eof;
    do { 
      enkf_sched_fscanf_alloc_nodes(enkf_sched , stream , joblist , use_lsf , &at_eof);
    } while (!at_eof);
    
    fclose( stream );
  }
  {
    enkf_sched_node_type * last_node = vector_get_last( enkf_sched->nodes );
    *last_report_step = last_node->report_step2;
  }
  enkf_sched_verify__(enkf_sched);
  return enkf_sched;
}



void enkf_sched_fprintf(const enkf_sched_type * enkf_sched , FILE * stream) {
  int i;
  for (i=0; i < vector_get_size( enkf_sched->nodes ); i++) 
    enkf_sched_node_fprintf(vector_iget_const( enkf_sched->nodes , i) , stream );
  
}



int enkf_sched_get_schedule_num_reports(const enkf_sched_type * enkf_sched) {
  return enkf_sched->schedule_num_reports;
}


int enkf_sched_get_last_report(const enkf_sched_type * enkf_sched) {
  return enkf_sched->last_report;
}

int enkf_sched_get_num_nodes(const enkf_sched_type * enkf_sched) {
  return vector_get_size( enkf_sched->nodes );
}



/**
   This function takes a report number, and returns the index of
   enkf_sched_node which contains (in the half-open interval: [...>)
   this report number. The function will abort if the node can be found.
*/
int enkf_sched_get_node_index(const enkf_sched_type * enkf_sched , int report_step) {
  if (report_step < 0 || report_step >= enkf_sched->last_report) {
    printf("Looking for report_step:%d \n", report_step);
    enkf_sched_fprintf(enkf_sched , stdout);
    util_abort("%s: could not find it ... \n",__func__);
    return -1;
  } else {
    int index = 0;
    while (1) {
      const enkf_sched_node_type * node = vector_iget_const(enkf_sched->nodes , index);
      if (node->report_step1 <= report_step && node->report_step2 > report_step)
	break;
      index++;
    }
    return index;
  }
}


const enkf_sched_node_type * enkf_sched_iget_node(const enkf_sched_type * enkf_sched , int index) {
  return vector_iget_const( enkf_sched->nodes , index );
}
