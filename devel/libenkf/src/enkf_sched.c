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
  int    size;                                /* NUmber of elements in the node_lost */
  enkf_sched_node_type ** node_list;


  int                      schedule_num_reports; /* The number of DATES / TSTEP keywords in the currently active SCHEDULE file. */
  int                      last_report;          /* The last report_step in the enkf_sched instance - can be greater than schedule_num_reports .*/
  const ext_joblist_type * joblist; 
  const stringlist_type  * std_forward_model;    /* Does not take ownership of this - i.e. it must be freed by the calling scope. */ 
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
  node->forward_model = (stringlist_type *) forward_model;
  return node;
}



void enkf_sched_node_get_data(const enkf_sched_node_type * node , int * report_step1 , int * report_step2 , int * report_stride , bool * enkf_on , stringlist_type ** forward_model) {
  *report_step1  = node->report_step1;
  *report_step2  = node->report_step2;
  *report_stride = node->report_stride;
  *enkf_on       = node->enkf_active;
  *forward_model = node->forward_model;
}

static void enkf_sched_node_free(enkf_sched_node_type * node , const stringlist_type * std_forward_model) {
  if (node->forward_model != std_forward_model)
    stringlist_free(node->forward_model);
  free(node);
}


static void enkf_sched_node_fprintf(const enkf_sched_node_type * node , const stringlist_type * std_forward_model , FILE * stream) {
  if (node->enkf_active)
    fprintf(stream , "%4d   %4d   %s   %3d   ",node->report_step1 , node->report_step2 , "ON " , node->report_stride);
  else
    fprintf(stream , "%4d   %4d   %s   %3d   ",node->report_step1 , node->report_step2 , "OFF" , node->report_stride);

  if (node->forward_model != std_forward_model)
    stringlist_fprintf(node->forward_model , stream);
  else
    fprintf(stream, "*");

  stringlist_fprintf(node->forward_model , stream);

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

   * If no stride is found, a default stride is used. If the value of
     enkf_update is on, the default stride equals 1, otherwise it
     equals report_step2 - report_step1.

   * If no forward_model is found, the default forward model is used.

   * If the stream is positioned at an empty line NULL is returned. No
     comments (at present).
*/


static enkf_sched_node_type * enkf_sched_node_fscanf_alloc(FILE * stream, stringlist_type * default_forward_model, const ext_joblist_type * joblist , bool * at_eof) {
  enkf_sched_node_type * sched_node = NULL;
  char ** token_list;
  bool enkf_active = false; /* Compiler shut up */
  int report_step1 , report_step2, report_stride;
  int tokens;
  stringlist_type * forward_model = default_forward_model;

  char * line = util_fscanf_alloc_line(stream , at_eof);
  if (line != NULL) {
    util_split_string(line , " \t" , &tokens , &token_list);
    if (tokens >= 3) {
      /*util_abort("%s: fatal error when parsing line:\'%s\' - must have at least 3 tokens \n",__func__ , line);*/
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
	  model_length = tokens - model_start;
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
	printf("Har speseill forward modell \n");
	for (i = 0; i < argc; i++)
	  if (!ext_joblist_has_job(joblist , stringlist_iget(forward_model , i)))
	    util_abort("%s: the forward job:%s has not been installed.\n",__func__ , stringlist_iget(forward_model , i));
      }
    }
    util_free_stringlist(token_list , tokens);
    free(line);
  } 
  return sched_node;
}


/*****************************************************************/


/* static enkf_sched_node_type * enkf_sched_node_copyc(const enkf_sched_node_type * src , const stringlist_type * std_forward_model) { */
/*   enkf_sched_node_type *new; */

/*   if (src->forward_model == std_forward_model) */
/*     new = enkf_sched_node_alloc(src->report_step1 , src->report_step2 , src->report_stride , src->enkf_active , std_forward_model); */
/*   else  */
/*     new = enkf_sched_node_alloc(src->report_step1 , src->report_step2 , src->report_stride , src->enkf_active , stringlist_alloc_deep_copy(src->forward_model)); */

/*   return new; */
/* } */

/* /\**  */
/*     Inserts the new node at index location 'index'. */
/*  *\/ */

/* static void enkf_sched_iadd_node(enkf_sched_type * enkf_sched, int index, enkf_sched_node_type *node) { */
/*   if (index < 0 || index > enkf_sched->size) */
/*     util_abort("%s: index:%d invlid. Valid range: [0,%d] \n",__func__ , index ,  enkf_sched->size); */
/*   { */
/*     int new_size = enkf_sched->size + 1; */
/*     enkf_sched_node_type ** new_list = util_malloc(new_size * sizeof * new_list , __func__); */
/*     memcpy(new_list            ,  enkf_sched->node_list        , index * sizeof * new_list);  /\* Copying everything _before_ index *\/ */
/*     memcpy(&new_list[index+1]  , &enkf_sched->node_list[index] , (new_size - index) * sizeof * new_list);  /\* Copying everything _after_ index *\/ */
/*     new_list[index]        = node; */
/*     enkf_sched->node_list = new_list; */
/*     enkf_sched->size      = new_size; */
/*     free( enkf_sched->node_list ); */
/*     enkf_sched->last_report = util_int_max(enkf_sched->last_report , node->report_step2); */
/*   } */
/* } */



/* /\**  */
/*     Removes element number 'index' rom the node_list  */
/* *\/  */

/* static void enkf_sched_idel_node(enkf_sched_type * enkf_sched , int index) { */
/*   if (index < 0 || index >= enkf_sched->size) */
/*     util_abort("%s: index:%d invlid. Valid range: [0,%d) \n",__func__ , index ,  enkf_sched->size); */

/*   { */
/*     int new_size = enkf_sched->size - 1; */
/*     enkf_sched_node_type ** new_list = util_malloc(new_size * sizeof * new_list , __func__); */
/*     memcpy(new_list          ,  enkf_sched->node_list          , index * sizeof * new_list);  /\* Copying everything _before_ index *\/ */
/*     memcpy(&new_list[index]  , &enkf_sched->node_list[index+1] , (new_size - index) * sizeof * new_list);  /\* Copying everything _after_ index *\/ */
/*     free( enkf_sched->node_list ); */
/*     enkf_sched->node_list = new_list; */
/*     enkf_sched->size      = new_size; */
/*   } */
/* } */




static void enkf_sched_verify_list__(const enkf_sched_type * enkf_sched) {
  int index;
  for (index = 0; index < (enkf_sched->size - 1); index++) {
    /*int report1      = enkf_sched->node_list[index]->report_step1;*/
    int report2      = enkf_sched->node_list[index]->report_step2;
    int next_report1 = enkf_sched->node_list[index + 1]->report_step1;
    /*int next_report2 = enkf_sched->node_list[index + 1]->report_step2;*/
    
    if (report2 != next_report1) {
      enkf_sched_fprintf(enkf_sched , stdout);
      util_abort("%s - abort \n",__func__);
    }
  }
}



static void enkf_sched_free_nodelist( enkf_sched_type * enkf_sched) {
  int i;
    for (i=0; i < enkf_sched->size; i++)
    enkf_sched_node_free(enkf_sched->node_list[i] , enkf_sched->std_forward_model);
  free(enkf_sched->node_list);
  enkf_sched->size = 0;
}

void enkf_sched_free( enkf_sched_type * enkf_sched) {
  enkf_sched_free_nodelist( enkf_sched );
}





/**
   When we are adding a new node with two values of report_step1 and
   report_step2 there are many different possibilities:

   1. report_step > last_report: In this case we abort, because there
      will be a report interval [last_report , report_step> where we
      do not know what to do.

  
*/

static void enkf_sched_add_node(enkf_sched_type * enkf_sched , enkf_sched_node_type * new_node) {
  if (enkf_sched->size == 0) {
    enkf_sched->size = 1;
    enkf_sched->node_list    = util_malloc(sizeof * enkf_sched->node_list , __func__);
    enkf_sched->node_list[0] = new_node;
    enkf_sched->last_report  = new_node->report_step2;
  } else {
    int new_report1 = new_node->report_step1;
    int new_report2 = new_node->report_step2;
    if (new_report1 > enkf_sched->last_report) 
      util_abort("%s: going to far: new_report1:%d  last-report:%d \n",__func__ , new_report1 , enkf_sched->last_report );
    {
      enkf_sched_node_type ** new_node_list = util_malloc(sizeof * new_node_list * (enkf_sched->size + 3) , __func__); 
      int index , i;
      int new_length = util_int_max( new_node->report_step2 , enkf_sched->last_report); 

      int * index_map = util_malloc(sizeof * index_map * new_length , __func__);
      for (index = 0; index < enkf_sched->size; index++) {
	enkf_sched_node_type * node = enkf_sched->node_list[index];
	int report1 = node->report_step1;
	int report2 = node->report_step2;;
	for (i = report1; i < report2; i++)
	  index_map[i] = index;
      }

      for (i = new_report1; i < new_report2; i++)
	index_map[i] = enkf_sched->size; /* The new addition */
      
      {
	const stringlist_type * forward_model;
	int new_global_index = 0;
	int report1 = 0;
	int report2, report_index; 

	while (report1 < new_length) {
	  int index    = index_map[report1];
	  report_index = report1;
	  
	  while( report_index < new_length && index_map[report_index] == index)
	    report_index++;
	  
	  report2 = report_index;

	  if (index < enkf_sched->size) {
	    enkf_sched_node_type * node  = enkf_sched->node_list[index];
	    if (node->forward_model == enkf_sched->std_forward_model)
	      forward_model = enkf_sched->std_forward_model;	    
	    else 
	      forward_model = stringlist_alloc_deep_copy(node->forward_model);
	    new_node_list[new_global_index] = enkf_sched_node_alloc(report1 , report2 , node->report_stride , node->enkf_active ,forward_model);
	  } else 
	    new_node_list[new_global_index] = new_node;
	  
	  new_global_index++;
	  report1 = report_index;
	}

	enkf_sched_free_nodelist(enkf_sched);
	enkf_sched->node_list = new_node_list;
	enkf_sched->size = new_global_index;
      }
      free(index_map);
    }
  }
  enkf_sched_verify_list__(enkf_sched);
  enkf_sched->last_report = util_int_max(enkf_sched->last_report , new_node->report_step2);
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

void enkf_sched_random_test(enkf_sched_type * enkf_sched) {
  const int N = 100;
  int i;

  for (i=0; i < N; i++) {
    int r1 = random() % enkf_sched->last_report;
    int r2 = r1 + random() % 25;
    enkf_sched_node_type * node = enkf_sched_node_alloc(r1 , r2 , 1 , true , enkf_sched->std_forward_model);
    enkf_sched_add_node(enkf_sched , node);
    printf("%6d/%d  %d -> %d OK",i,N,r1,r2);
  }
}
  

/**
   This functions parses a config file, and builds a enkf_sched_type *
   instance from it. If the filename argument is NULL a default
   enkf_sched_type instance is allocated.
*/

enkf_sched_type * enkf_sched_fscanf_alloc(const char * enkf_sched_file , const sched_file_type * sched_file , const ext_joblist_type * joblist, const stringlist_type * default_forward_model) {
  
  enkf_sched_type * enkf_sched = enkf_sched_alloc_empty(sched_file , joblist ,default_forward_model);
  enkf_sched_set_default(enkf_sched);
  /*enkf_sched_random_test(enkf_sched);*/

  if (enkf_sched_file != NULL) {
    FILE * stream = util_fopen(enkf_sched_file , "r");
    enkf_sched_node_type * node;
    bool at_eof;
    do { 
      node = enkf_sched_node_fscanf_alloc(stream , (stringlist_type *) default_forward_model , joblist , &at_eof);
      if (node != NULL)
	enkf_sched_add_node(enkf_sched , node);
    } while (!at_eof);
    
    fclose( stream );
  }
  return enkf_sched;
}



void enkf_sched_fprintf(const enkf_sched_type * enkf_sched , FILE * stream) {
  int i;
  for (i=0; i < enkf_sched->size; i++)
    enkf_sched_node_fprintf(enkf_sched->node_list[i] , enkf_sched->std_forward_model , stream );
}



int enkf_sched_get_schedule_num_reports(const enkf_sched_type * enkf_sched) {
  return enkf_sched->schedule_num_reports;
}


int enkf_sched_get_num_nodes(const enkf_sched_type * enkf_sched) {
  return enkf_sched->size;
}


const enkf_sched_node_type * enkf_sched_iget_node(const enkf_sched_type * enkf_sched , int index) {
  if (index < 0 || index >= enkf_sched->size)
    util_abort("%s: Go fix your code - lazy bastartd ... \n",__func__);

  return enkf_sched->node_list[index];
}
