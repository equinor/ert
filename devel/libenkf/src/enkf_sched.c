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
#include <parser.h>
#include <enkf_defaults.h>

/**
   How long is the simulation?
   ===========================
   
   There are some many configuration variables and usage patterns
   control the length of the simulation that you can seriously get
   dizzy - they all come with their own potential for +/- 1
   misunderstandings. 

   User options:
   -------------

     SCHEDULE_FILE: This is basically the schedule file given by the
        user which controls the historical period. This schedule file
        is guranteed to provide all members with the same DATES /
        TSTEP keywords.

     PREDICTION_SCHEDULE_FILE: This is optional file which the user
        can supply for predictions. It will be parsed and appended to
        the internal sched_file instance. Observe that the
        PREDICTION_SCHEDULE_FILE can be per. member, hence for the
        prediction part there is no longer a guarantee that all
        member simulations are equally long.

     ENKF_SCHED_FILE: With this configuration option the user can have
        reasonably fine-grained control on the time steps in the
        simulation. The simulation will be controlled by an enkf_sched
        instance (implemented in this file), to supply a
        ENKF_SCHED_FILE is optional, if no file is supplied by the
        user a default enkf_sched instance is allocated.


   Usage patterns:
   ---------------
   




   
   Offset +/- 1 convention:
   ------------------------
   
   The main 'master' of this information is the sched_file instance,
   that provides the function sched_file_get_num_restart_files() which
   returns the total number of restart files from a full simulation:


      schedule_file.inc
      -----------------
      --- Simulation START 1. JAN 2000  

      DATES 
        1  'FEB' 2000  /
      /

      
      DATES
        1  'MAR' 2000 /
      /


      DATES
        1  'APR' 2000 /
      /
      END

   This schedule file produce the following restart files when a full
   simulation is run:

      Restart_file   |    Corresponding date
      --------------------------------------
      BASE.X0000     |       1. JAN  2000
      BASE.X0001     |       1. FEB  2000
      BASE.X0002     |       1. MAR  2000
      BASE.X0003     |       1. APR  2000
      --------------------------------------

   So in this case the function sched_file_get_num_restarts() will
   return 4, but the last valid restart file has number '0003'. The
   fundamental query in this functionality is to query the sched_file
   instance, and that will return 4 in the example shown above. On the
   other hand most of the API and user interface in this application
   is based on an inclusive upper limit, i.e. we translate

      "Total number of restart files: 4" ==> "Number of the last restart file: 3"

   This translation is done 'immediately' the sched_file routine
   returns 4, and that should be immediately converted to three.



   Observe that the functions in this file are run - not at system
   bootstrap, but when starting a simulation.
*/



struct enkf_sched_node_struct {
  int    	       report_step1;
  int    	       report_step2;
  bool                 enkf_active;
  forward_model_type * forward_model;   /* Will be different from NULL only for updates which have a 'special' forward model. */
};





struct enkf_sched_struct {
  vector_type    *nodes;                  /* Vector consisting of enkf_sched_node_type instances. */
  int             last_report;            /* The last report_step in this enkf_sched instance - internal convenience variable. */
};




static enkf_sched_node_type * enkf_sched_node_alloc(int report_step1 , int report_step2 , bool enkf_active , forward_model_type * forward_model) {
  enkf_sched_node_type * node = util_malloc(sizeof * node , __func__);
  node->report_step1  = report_step1;
  node->report_step2  = report_step2;
  node->enkf_active   = enkf_active;
  /*
    if (forward_model != NULL) 
    util_abort("%s : Sorry - support for special forward models has been (temporarily) removed\n",__func__);
  */
  
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


static void  enkf_sched_fscanf_alloc_nodes(enkf_sched_type * enkf_sched , FILE * stream , const ext_joblist_type * joblist , bool statoil_mode , bool use_lsf , bool * at_eof) {
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
	    forward_model = forward_model_alloc( joblist , statoil_mode , use_lsf , DEFAULT_START_TAG , DEFAULT_END_TAG );
            forward_model_parse_init( forward_model , input_string );
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
      util_abort("%s report steps must be continous - there is a gap.\n",__func__);
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





static enkf_sched_type * enkf_sched_alloc_empty( ) {
  enkf_sched_type * enkf_sched      = util_malloc(sizeof * enkf_sched , __func__);
  enkf_sched->nodes                 = vector_alloc_new();  
  enkf_sched->last_report           = 0;
  return enkf_sched;
}



static void  enkf_sched_set_default(enkf_sched_type * enkf_sched , int last_history_restart , int abs_last_restart , run_mode_type run_mode) {
  enkf_sched_node_type * node;

  if (run_mode == ENKF_ASSIMILATION) {
    /* Default enkf: stride one - active at all report steps. */
    /* Have to explicitly add all these nodes. */
    int report_step;
    for (report_step = 0; report_step < last_history_restart; report_step++) {   
      node = enkf_sched_node_alloc(report_step , report_step + 1, true , NULL);
      enkf_sched_append_node(enkf_sched , node);
    }
    /* Okay we are doing assimilation and prediction in one go - fair enough. */
    
    if (abs_last_restart > last_history_restart) {
      /* We have prediction. */
      node = enkf_sched_node_alloc(last_history_restart , abs_last_restart , false , NULL);
      enkf_sched_append_node(enkf_sched , node);
    }
  } else {
    /* 
       experiment: Do the whole thing in two steps, 
       first the whole history, and then subsequently the prediction part (if there is any).
    */
    node = enkf_sched_node_alloc(0 , last_history_restart , false , NULL); 
    enkf_sched_append_node(enkf_sched , node);
    if (abs_last_restart > last_history_restart) {
      /* We have prediction. */
      node = enkf_sched_node_alloc(last_history_restart , abs_last_restart , false , NULL);
      enkf_sched_append_node(enkf_sched , node);
    }
  }
}




/**
   This functions parses a config file, and builds a enkf_sched_type *
   instance from it. If the filename argument is NULL a default
   enkf_sched_type instance is allocated.
*/

enkf_sched_type * enkf_sched_fscanf_alloc(const char * enkf_sched_file , int last_history_restart , int abs_last_restart , run_mode_type run_mode, const ext_joblist_type * joblist , bool statoil_mode , bool use_lsf) {
  enkf_sched_type * enkf_sched = enkf_sched_alloc_empty( );
  if (enkf_sched_file == NULL)
    enkf_sched_set_default(enkf_sched , last_history_restart , abs_last_restart , run_mode);
  else {
    FILE * stream = util_fopen(enkf_sched_file , "r");
    bool at_eof;
    do { 
      enkf_sched_fscanf_alloc_nodes(enkf_sched , stream , joblist , statoil_mode , use_lsf , &at_eof);
    } while (!at_eof);
    
    fclose( stream );
  }
  enkf_sched_verify__(enkf_sched);
  return enkf_sched;
}



void enkf_sched_fprintf(const enkf_sched_type * enkf_sched , FILE * stream) {
  int i;
  for (i=0; i < vector_get_size( enkf_sched->nodes ); i++) 
    enkf_sched_node_fprintf(vector_iget_const( enkf_sched->nodes , i) , stream );
  
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
