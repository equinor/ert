#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <util.h>
#include <gen_data.h>
#include <enkf_types.h>
#include <ecl_util.h>



/**
   This file implements a "general data type"; the data type is
   originally devised to load and organise seismic data from a forward
   simulation. 

   In contrast to the other enkf data types this type stores quite a
   lot of configuration information in the actual data object, and not
   in a config object. That is because much config information, like
   e.g. the length+++ is determined at run time when we load data
   produced by an external application.

   The same gen_data object can (in principle) be used in different
   ways through one simulation.
*/
   


#define  DEBUG
#define  TARGET_TYPE GEN_DATA
#include "enkf_debug.h"


struct gen_data_struct {
  DEBUG_DECLARE
  const  gen_data_config_type  * config;    /* Normal config object - mainly contains filename for remote load */
  ecl_type_enum                  ecl_type;  /* Type of data can be ecl_float_type || ecl_double_type - read at load time, 
					       can change from time-step to time-step.*/           
  int   			 size;      /* The number of elements. */
  bool  			 active;    /* Is the keyword currently active ?*/
  char                         * ext_tag;   /* A tag written by the external software which identifies the data - can be NULL */ 
  char                         * data;      /* Actual storage - will typically be casted to double or float on use. */
};



/*****************************************************************/


void gen_data_free(gen_data_type * gen_data) {
  util_safe_free(gen_data->data);
  free(gen_data);
}


/**
   Would prefer that this function was not called at all if the
   gen_data keyword does not (currently) hold data. The current
   implementation will leave a very small file without data.
*/

void gen_data_fwrite(const gen_data_type * gen_data , FILE * stream) {
  DEBUG_ASSERT(gen_data)
  enkf_util_fwrite_target_type(stream , GEN_DATA);
  util_fwrite_bool(gen_data->active , stream);
  if (gen_data->active) {
    util_fwrite_int(gen_data->size , stream);
    util_fwrite_int(gen_data->ecl_type , stream);
    util_fwrite_compressed(gen_data->data , gen-data->size * ecl_util_get_sizeof_ctype(gen_data->ecl_type) , stream);
  }
}


