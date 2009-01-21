#include <stdlib.h>
#include <stdbool.h>
#include <util.h>
#include <string.h>
#include <stdio.h>
#include <enkf_sched.h>
#include <stringlist.h>
#include <ext_joblist.h>
#include <forward_model.h>



struct enkf_sched_node_struct {
  int    	       report_step1;
  int    	       report_step2;
  int                  report_stride; 
  bool                 enkf_active;
  forward_model_type * forward_model;   /* Will mostly b eNULL */
};





struct enkf_sched_struct {
  int    alloc_size;                                   /* Allocated size - internal variable. */  
  int    size;                                         /* Number of elements in the node_list */
  enkf_sched_node_type          ** node_list;
  
  
  int                          	 schedule_num_reports; /* The number of DATES / TSTEP keywords in the currently active SCHEDULE file. */
  int                          	 last_report;          /* The last report_step in the enkf_sched instance - can be greater than schedule_num_reports .*/
};



static enkf_sched_node_type * enkf_sched_node_alloc(int report_step1 , int report_step2 , int report_stride , bool enkf_active , forward_model_type * forward_model) {
  enkf_sched_node_type * node = util_malloc(sizeof * node , __func__);
  node->report_step1  = report_step1;
  node->report_step2  = report_step2;
  node->report_stride = report_stride;
  node->enkf_active   = enkf_active;
  if (forward_model != NULL) 
    util_abort("%s : Sorry - support for special forward models has been (temporarily) removed\n",__func__);
  
  node->forward_model = forward_model; 
  return node;
}



void enkf_sched_node_get_data(const enkf_sched_node_type * node , int * report_step1 , int * report_step2 , int * report_stride , bool * enkf_on , forward_model_type ** forward_model) {
  *report_step1    = node->report_step1;
  *report_step2    = node->report_step2;
  *report_stride   = node->report_stride;
  *enkf_on         = node->enkf_active;
  *forward_model   = node->forward_model;
}



static void enkf_sched_node_free(enkf_sched_node_type * node) {
  if (node->forward_model != NULL)
    forward_model_free( node->forward_model);

  free(node);
}


static void enkf_sched_node_fprintf(const enkf_sched_node_type * node , FILE * stream) {
  if (node->enkf_active)
    fprintf(stream , "%4d   %4d   %s   %3d   ",node->report_step1 , node->report_step2 , "ON " , node->report_stride);
  else
    fprintf(stream , "%4d   %4d   %s   %3d   ",node->report_step1 , node->report_step2 , "OFF" , node->report_stride);
  
  if (node->forward_model != NULL)
    forward_model_fprintf( node->forward_model  , stream );
  else
    fprintf(stream, "*");
  
  fprintf(stream , "\n");
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

   * If no stride is found, a default stride is used - the default
     stride is report_step2 - report_step1 (i.e. the whole thing in
     one go.)

   * If no forward_model is found, the default forward model is used.

   * If the stream is positioned at an empty line NULL is returned. No
     comments (at present).
*/


static enkf_sched_node_type * enkf_sched_node_fscanf_alloc(FILE * stream , const ext_joblist_type * joblist , bool use_lsf , bool * at_eof) {
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
	if (strcmp(token_list[2] , "ON") == 0) {
	  enkf_active = true;
	  report_stride = report_step2 - report_step1;
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
	  model_length = tokens - model_start;
	  if (model_length > 0) {
	    char * input_string = util_alloc_joined_string( (const char **) &token_list[model_start] , model_length , " ");
	    forward_model = forward_model_alloc( input_string , joblist , use_lsf );
	    free( input_string );
	  }
	}
      } else
	util_abort("%s: failed to parse %s and %s as integers\n",__func__ , token_list[0] , token_list[1]);

      sched_node = enkf_sched_node_alloc(report_step1 , report_step2 , report_stride , enkf_active , forward_model);
    }
    util_free_stringlist(token_list , tokens);
    free(line);
  } 
  return sched_node;
}


/*****************************************************************/



static void enkf_sched_verify__(const enkf_sched_type * enkf_sched) {
  int index;
  /* Verify that first node starts at zero. */
  if (enkf_sched->node_list[0]->report_step1 != 0)
    util_abort("%s: must start at report-step 0 \n",__func__);

  for (index = 0; index < (enkf_sched->size - 1); index++) {
    int report1      = enkf_sched->node_list[index]->report_step1;
    int report2      = enkf_sched->node_list[index]->report_step2;
    int next_report1 = enkf_sched->node_list[index + 1]->report_step1;
    
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
    index = enkf_sched->size - 1;
    int report1      = enkf_sched->node_list[index]->report_step1;
    int report2      = enkf_sched->node_list[index]->report_step2;
    if (report2 <= report1)
      util_abort("%s: enkf_sched step of zero/negative length:%d - %d - that is not allowed \n",__func__ , report1 , report2);
      
  }

}



static void enkf_sched_free_nodelist( enkf_sched_type * enkf_sched) {
  int i;
  for (i=0; i < enkf_sched->size; i++)
    enkf_sched_node_free(enkf_sched->node_list[i]);
  free(enkf_sched->node_list);
}


void enkf_sched_free( enkf_sched_type * enkf_sched) {
  enkf_sched_free_nodelist( enkf_sched );
  free( enkf_sched );
}



static void enkf_sched_realloc(enkf_sched_type * enkf_sched, int new_alloc_size) {
  int i;
  enkf_sched->node_list = util_realloc(enkf_sched->node_list , new_alloc_size * sizeof * enkf_sched->node_list , __func__);
  for (i=enkf_sched->alloc_size; i < new_alloc_size; i++)
    enkf_sched->node_list[i] = NULL;
  enkf_sched->alloc_size = new_alloc_size;
}



static void enkf_sched_append_node(enkf_sched_type * enkf_sched , enkf_sched_node_type * new_node) {
  if (enkf_sched->size == enkf_sched->alloc_size)
    enkf_sched_realloc(enkf_sched , 2 * (enkf_sched->alloc_size + 1));

  enkf_sched->node_list[enkf_sched->size] = new_node;
  enkf_sched->last_report  = new_node->report_step2;
  enkf_sched->size++;
}



static enkf_sched_type * enkf_sched_alloc_empty( int num_restart_files) {
  enkf_sched_type * enkf_sched     = util_malloc(sizeof * enkf_sched , __func__);
  enkf_sched->node_list 	   = NULL;
  enkf_sched->size      	   = 0;       
  enkf_sched->alloc_size           = 0;
  enkf_sched->schedule_num_reports = num_restart_files - 1;
  enkf_sched->last_report          = 0;
  return enkf_sched;
}



static void  enkf_sched_set_default(enkf_sched_type * enkf_sched ) {
  enkf_sched_node_type * node = enkf_sched_node_alloc(0 , enkf_sched->schedule_num_reports , 1 , true , NULL);
  enkf_sched_append_node(enkf_sched , node);
}



/**
   This functions parses a config file, and builds a enkf_sched_type *
   instance from it. If the filename argument is NULL a default
   enkf_sched_type instance is allocated.
*/

enkf_sched_type * enkf_sched_fscanf_alloc(const char * enkf_sched_file , int num_restart_files , const ext_joblist_type * joblist , bool use_lsf) {
  enkf_sched_type * enkf_sched = enkf_sched_alloc_empty(num_restart_files);
  if (enkf_sched_file == NULL)
    enkf_sched_set_default(enkf_sched);
  else {
    FILE * stream = util_fopen(enkf_sched_file , "r");
    enkf_sched_node_type * node;
    bool at_eof;
    do { 
      node = enkf_sched_node_fscanf_alloc(stream , joblist , use_lsf , &at_eof);
      if (node != NULL)
	enkf_sched_append_node(enkf_sched , node);
    } while (!at_eof);
    
    fclose( stream );
  }
  enkf_sched_verify__(enkf_sched);
  return enkf_sched;
}



void enkf_sched_fprintf(const enkf_sched_type * enkf_sched , FILE * stream) {
  int i;
  for (i=0; i < enkf_sched->size; i++)
    enkf_sched_node_fprintf(enkf_sched->node_list[i] , stream );
}



int enkf_sched_get_schedule_num_reports(const enkf_sched_type * enkf_sched) {
  return enkf_sched->schedule_num_reports;
}


int enkf_sched_get_last_report(const enkf_sched_type * enkf_sched) {
  return enkf_sched->last_report;
}

int enkf_sched_get_num_nodes(const enkf_sched_type * enkf_sched) {
  return enkf_sched->size;
}



/**
   This function takes a report number, and returns the index of
   enkf_sched_node which contains (in the half-open interval: [...>)
   this report number. 
   
   The function will return -1 if the report number can not be
   found. 
*/
int enkf_sched_get_node_index(const enkf_sched_type * enkf_sched , int report_step) {
  if (report_step < 0 || report_step >= enkf_sched->last_report)
    return -1;
  else {
    int index = 0;
    while (1) {
      const enkf_sched_node_type * node = enkf_sched->node_list[index];
      if (node->report_step1 <= report_step && node->report_step2 > report_step)
	break;
      index++;
    }
    return index;
  }
}


const enkf_sched_node_type * enkf_sched_iget_node(const enkf_sched_type * enkf_sched , int index) {
  if (index < 0 || index >= enkf_sched->size)
    util_abort("%s: Go fix your code - lazy bastartd ... \n",__func__);

  return enkf_sched->node_list[index];
}
