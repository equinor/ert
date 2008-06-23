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


void gen_data_realloc_data(gen_data_type * gen_data) {
  gen_data->data = util_realloc(gen-data->data , gen-data->size * ecl_util_get_sizeof_ctype(gen_data->ecl_type) , __func__);
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


void gen_data_fread(gen_data_type * gen_data , FILE * stream) {
  DEBUG_ASSERT(gen_data)
  enkf_util_fread_target_type(stream , GEN_DATA);
  util_fread_bool(gen_data->active , stream);
  if (gen_data->active) {
    util_fread_int(gen_data->size , stream);
    util_fread_int(gen_data->size , stream);
    gen_data_realloc_data(gen_data);
    util_fread_compressed(gen_data->data , stream);
  }
}

static void gen_data_deactivate(gen_data_type * gen_data) {
  if (gen_data->active) {
    gen_data->active = false;
    gen_data->size   = 0;
    gen_data_free_data( gen_data );
  }
}


void gen_data_ecl_read(gen_data_type * gen_data , const char * run_path , const char * ecl_base , const ecl_sum_type * ecl_sum , int report_step) {
  DEBUG_ASSERT(gen_data)
  {
    const gen_data_config_type * config = gen_data->config;
    /*
      At this stage we could ask the config object both whether the
      keyword is active at this stage, and for a spesific keyword to
      match??
    */
    char * ecl_file = util_alloc_full_path(run_path , gen_data_config_get_eclfile(config));
    if (util_file_exists(ecl_file)) {
      FILE * stream = util_fopen(ecl_file , "r");
      
      fclose(stream);
    } else 
      gen_data_deactivate(gen_data);

    free(ecl_file);
  }
}
