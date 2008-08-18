#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <enkf_main.h>
#include <enkf_sched.h>
#include <void_arg.h>
#include <void_arg.h>
#include <field.h>
#include <enkf_state.h>
#include <enkf_config.h>


/** 
    This file implements various small utility functions for the (text
    based) EnKF user interface.
*/



/**
   This functions displays the user with a prompt, and reads 'A' or
   'F' (lowercase OK), to check whether the user is interested in the
   forecast or the analyzed state.
*/


state_enum enkf_ui_util_scanf_state(const char * prompt) {
  char analyzed_string[64];
  bool OK;
  state_enum state;
  do {
    OK = true;
    printf("%s",prompt);
    scanf("%s" , analyzed_string);
    if (strlen(analyzed_string) == 1) {
      char c = toupper(analyzed_string[0]);
      if (c == 'A')
	state = analyzed;
      else if (c == 'F')
	state = forecast;
      else
	OK = false;
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

   The values are returned by reference. The keyword is checked for
   existence; but it is not checked whether the report_step actually
   exists.
*/

void enkf_ui_util_scanf_parameter(const enkf_config_type * config , char ** key , int * report_step , state_enum * state , int * iens) {
  char kw[256];
  bool kw_exists = false;
  do {
    printf("Keyword ==================> "); 
    scanf("%s" , kw);
    kw_exists = enkf_config_has_key(config , kw);
    if (kw_exists) {
      *report_step = util_scanf_int("Report step ==============> ");
      *state       = enkf_ui_util_scanf_state("Analyzed/forecast [A|F] ==> ");
      *iens        = util_scanf_int("Ensemble member [0 based]=> ");
    }
  } while (!kw_exists);
  *key = util_alloc_string_copy(kw);
}
