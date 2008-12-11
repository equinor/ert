#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <enkf_main.h>
#include <enkf_sched.h>
#include <enkf_node.h>
#include <arg_pack.h>
#include <arg_pack.h>
#include <field.h>
#include <field_config.h>
#include <enkf_state.h>
#include <ensemble_config.h>
#include <enkf_types.h>


/** 
    This file implements various small utility functions for the (text
    based) EnKF user interface.
*/



/**
   This functions displays the user with a prompt, and reads 'A' or
   'F' (lowercase OK), to check whether the user is interested in the
   forecast or the analyzed state.
*/


state_enum enkf_ui_util_scanf_state(const char * prompt, int prompt_len, bool accept_both) {
  char analyzed_string[64];
  bool OK;
  state_enum state;
  do {
    OK = true;
    util_printf_prompt(prompt , prompt_len , '=' , "=> ");
    scanf("%s" , analyzed_string);
    if (strlen(analyzed_string) == 1) {
      char c = toupper(analyzed_string[0]);
      if (c == 'A')
	state = analyzed;
      else if (c == 'F')
	state = forecast;
      else {
	if (accept_both) {
	  if (c == 'B') 
	    state = both;
	  else
	    OK = false;
	} else
	  OK = false;
      }
    } else
      OK = false;
  } while ( !OK );
  return state;
}



/**
   Very simple function which is in interactive functions. Used to
   query the user:

     - Key identifying a field.
     - An integer report step.
     - Whether we are considering the analyzed state or the forecast.

   The config_node is returned, and in addition the report_step, iens
   and analysis_state are returned by reference. It is OK the pass in
   NULL for these pointers; in that case the user is not queried for
   these values.

   The keyword is checked for existence; but it is not checked whether
   the report_step actually exists. If impl_type == INVALID, any
   implementation type will be accepted, otherwise we loop until the
   keyword is of type impl_type.
*/

const enkf_config_node_type * enkf_ui_util_scanf_parameter(const ensemble_config_type * config , int prompt_len , bool accept_both , enkf_impl_type impl_type ,  enkf_var_type var_type , int * report_step , state_enum * state , int * iens) {
  char kw[256];
  bool OK;
  const enkf_config_node_type * config_node;
  do {
    OK = true;
    util_printf_prompt("Keyword" , prompt_len , '=' , "=> ");
    scanf("%s" , kw);
    if (ensemble_config_has_key(config , kw)) {
      config_node = ensemble_config_get_node(config , kw);
      
      if (impl_type != INVALID) 
	if (enkf_config_node_get_impl_type(config_node) != impl_type) 
	  OK = false;
      
      if (var_type != invalid)
	if (enkf_config_node_get_var_type(config_node) != var_type) 
	  OK = false;
      
      /*if (!OK) 
	fprintf(stderr,"Error: %s is of type:\"%s\" - you must give a keyword of type: \"%s\" \n",kw,enkf_types_get_impl_name(enkf_config_node_get_impl_type(config_node)) , enkf_types_get_impl_name(impl_type));
      */
      if (!OK)
	fprintf(stderr,"ERROR: %s has wrong type \n",kw);
      else {
	if (report_step != NULL) *report_step = util_scanf_int("Report step" , prompt_len);
	if (state != NULL) {
	  
	  if (accept_both)  /* It does not make sense to plot both forecast and updated for parameters.*/
	    if (!(enkf_config_node_get_var_type(config_node) & (dynamic_result + dynamic_state)))
	      accept_both = false;

	  if (accept_both)
	    *state = enkf_ui_util_scanf_state("Analyzed/forecast [A|F|B]" , prompt_len , true);
	  else
	    *state = enkf_ui_util_scanf_state("Analyzed/forecast [A|F]" , prompt_len , false);
	}
      }
      if (iens != NULL)  *iens = util_scanf_int_with_limits("Ensemble member" , prompt_len , 0 , ensemble_config_get_size(config) - 1);
    } else OK = false;
  } while (!OK);
  return config_node;
}


/**
   Present the user with the queries:
   
      First ensemble member ==>
      Last ensemble member ===>
  
    It then allocates (bool *) pointer [0..ens_size-1], where the
    interval gven by the user is true (i.e. actve), and the rest is
    false. It s the responsiibility of the calling scope to free this.
*/


bool * enkf_ui_util_scanf_alloc_iens_active(int ens_size, int prompt_len , int * _iens1 , int * _iens2) {
  bool * iactive = util_malloc(ens_size * sizeof * iactive , __func__);
  int iens1 = util_scanf_int_with_limits("First ensemble member" , prompt_len , 0 , ens_size - 1);
  int iens2 = util_scanf_int_with_limits("Last ensemble member" , prompt_len , iens1 , ens_size - 1);
  int iens;

  for (iens = 0; iens < ens_size; iens++) 
    iactive[iens] = false;

  for (iens = iens1; iens <= iens2; iens++) 
    iactive[iens] = true;


  *_iens1 = iens1;
  *_iens2 = iens2;
  return iactive;
}



/**
   Presents the reader with a prompt, and reads a string containing
   two integers separated by a character(s) in the set: " ,-:".

   Will not return before the user has actually presented a valid
   string.
*/

  
void enkf_ui_util_scanf_iens_range(int ens_size , int prompt_len , int * iens1 , int * iens2) {
  char * prompt = util_alloc_sprintf("Ensemble members (0 - %d)" , ens_size - 1);
  bool OK = false;

  util_printf_prompt(prompt , prompt_len , '=' , "=> ");
  
  while (!OK) {
    char * input = util_alloc_stdin_line();
    const char * current_ptr = input;
    OK = true;

    current_ptr = util_parse_int(current_ptr , iens1 , &OK);
    current_ptr = util_skip_sep(current_ptr , " ,-:" , &OK);
    current_ptr = util_parse_int(current_ptr , iens2 , &OK);
    
    if (!OK) 
      printf("Failed to parse two integers from: \"%s\". Example: \"0 - 19\" to get the 20 first members.\n",input);
    free(input);
  }
  free(prompt);
}


void enkf_ui_util_scanf_report_steps(int last_report , int prompt_len , int * step1 , int * step2) {
  char * prompt = util_alloc_sprintf("Report steps (0 - %d)" , last_report);
  bool OK = false;

  util_printf_prompt(prompt , prompt_len , '=' , "=> ");
  
  while (!OK) {
    char * input = util_alloc_stdin_line();
    const char * current_ptr = input;
    OK = true;

    current_ptr = util_parse_int(current_ptr , step1 , &OK);
    current_ptr = util_skip_sep(current_ptr , " ,-:" , &OK);
    current_ptr = util_parse_int(current_ptr , step2 , &OK);
    
    if (!OK) 
      printf("Failed to parse two integers from: \"%s\". Example: \"0 - 19\" to get the 20 first report steps.\n",input);
    free(input);
  }
  free(prompt);
}



/**
   Similar to enkf_ui_util_scanf_alloc_iens_active(), but based on report steps.
*/

bool * enkf_ui_util_scanf_alloc_report_active(const enkf_sched_type * enkf_sched , int prompt_len) {
  const int last_step = enkf_sched_get_last_report(enkf_sched);
  bool * iactive = util_malloc((last_step + 1) * sizeof * iactive , __func__);
  int step1 = util_scanf_int_with_limits("First report step" , prompt_len , 0 , last_step);
  int step2 = util_scanf_int_with_limits("Last report step" , prompt_len , step1 , last_step);
  int step;

  for (step = 0; step <= last_step; step++) 
    iactive[step] = false;

  for (step = step1; step <= step2; step++) 
    iactive[step] = true;

  return iactive;
}


/** 
    This functions reads i,j,k and returns them be reference; if the
    reference pointer is NULL, that coordinate is skipped. I.e.

    enkf_ui_util_scanf_ijk__(config , 100 , &i , &j , NULL);

    Will read i and j. If your are interested in all three coordinates
    you should use enkf_ui_util_scanf_ijk() which has a more flexible
    parser.
*/


void enkf_ui_util_scanf_ijk__(const field_config_type * config, int prompt_len , int *i , int *j , int *k) {
  int nx,ny,nz;

  field_config_get_dims(config , &nx , &ny , &nz);
  if (i != NULL) (*i) = util_scanf_int_with_limits("Give i-index" , prompt_len , 1 , nx) - 1;
  if (j != NULL) (*j) = util_scanf_int_with_limits("Give j-index" , prompt_len , 1 , ny) - 1;
  if (k != NULL) (*k) = util_scanf_int_with_limits("Give k-index" , prompt_len , 1 , nz) - 1;
}




/**
   The function reads ijk, but it returns a global 1D index. Observe
   that the user is supposed to enter an index starting at one - whichs
   is immediately shifted down to become zero based.

   The function will loop until the user has entered ijk corresponding
   to an active cell.
*/
   
int enkf_ui_util_scanf_ijk(const field_config_type * config, int prompt_len) {
  int global_index;
  field_config_scanf_ijk(config , true , "Give (i,j,k) indices" , prompt_len , NULL , NULL , NULL , &global_index);
  return global_index;
}









/**
   This function runs through all the report steps [step1:step2] for
   member iens, and gets the value of the cell 'get_index'. Current
   implementation assumes that the config_node/node comination are of
   field type - this should be generalized to use the enkf_node_iget()
   function.

   The value is returned (by reference) in y, and the corresponding
   time (currently report_step) is returned in 'x'.
*/
   

void enkf_ui_util_get_time(enkf_fs_type * fs , const enkf_config_node_type * config_node, enkf_node_type * node , state_enum analysis_state , int get_index , int step1 , int step2 , int iens , double * x , double * y ) {
  const char * key = enkf_config_node_get_key_ref(config_node);
  int report_step;
  int index = 0;
  for (report_step = step1; report_step <= step2; report_step++) {
    
    if (analysis_state & forecast) {
      if (enkf_fs_has_node(fs , config_node , report_step , iens , forecast)) {
	enkf_fs_fread_node(fs , node , report_step , iens , forecast); {
	  const field_type * field = enkf_node_value_ptr( node );
	  y[index] = field_iget_double(field , get_index);
	}
      } else {
	fprintf(stderr," ** Warning field:%s is missing for member,report: %d,%d \n",key  , iens , report_step);
	y[index] = -1;
      }
      x[index] = report_step;
      index++;
    }
    
    
    if (analysis_state & analyzed) {
      if (enkf_fs_has_node(fs , config_node , report_step , iens , analyzed)) {
	enkf_fs_fread_node(fs , node , report_step , iens , analyzed); {
	  const field_type * field = enkf_node_value_ptr( node );
	  y[index] = field_iget_double(field , get_index);
	}
      } else {
	fprintf(stderr," ** Warning field:%s is missing for member,report: %d,%d \n",key , iens , report_step);
	y[index] = -1;
      }
      x[index] = report_step;
      index++;
    }
  }
}


int enkf_ui_util_scanf_report_step(const enkf_main_type * enkf_main , const char * prompt , int prompt_len) {
  const enkf_sched_type      * enkf_sched = enkf_main_get_enkf_sched(enkf_main);
  const int last_report                   = enkf_sched_get_last_report(enkf_sched);
  int report_step = util_scanf_int_with_limits(prompt , prompt_len , 0 , last_report);
}


