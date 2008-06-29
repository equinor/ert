#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <util.h>
#include <ecl_sum.h>
#include <enkf_types.h>
#include <enkf_util.h>
#include <fortio.h>
#include <ecl_util.h>
#include <gen_data_config.h>
#include <gen_data.h>



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
   
typedef enum {ascii_file , binary_C_file , binary_fortran_file} gen_data_file_type;


#define  DEBUG
#define  TARGET_TYPE GEN_DATA
#include "enkf_debug.h"


struct gen_data_struct {
  DEBUG_DECLARE
  const  gen_data_config_type  * config;      	      /* Thin config object - mainly contains filename for remote load */
  ecl_type_enum                  ecl_type;    	      /* Type of data can be ecl_float_type || ecl_double_type - read at load time, 
					      	         can change from time-step to time-step.*/           
  int   			 size;        	      /* The number of elements. */
  bool  			 active;      	      /* Is the keyword currently active ?*/
  char                         * ext_tag;     	      /* A tag written by the external software which identifies the data - can be NULL */ 
  char                         * data;        	      /* Actual storage - will be casted to double or float on use. */

  /*-----------------------------------------------------------------*/
  /* The variables below this line are only relevant when we are
     loading results from a forward simulation, and not actually part
     of the gen_data structure as such.
  */

  gen_data_file_type             file_type;   	      /* The type of file this is. */ 
  bool                           fortran_endian_flip; /* IFF file_type == binary_fortran_file this variable is correct - otherwise rubbish. */
  char                         * src_file;            /* Name of the src_file we are currently trying to load from. */
};




/*****************************************************************/
/**
Format of the gen_data files should be:
ASCII|BINARY
N:KEYWORD
DOUBLE|FLOAT
<SIZE>
d1
d2
d3
d4
d5
....
....

Observe the following: For the keyword you must specify the length of
the keyword as an integer before the actual keyword, separated from
the keyword with a ":" - OK that is butt ugly - no reason to scream
and shout :-)

I.e. to write an ascii file (with C):

   fprintf(stream, "ASCII\n");
   fprintf(stream,"%d:%s\n",strlen(keyword) , keyword);
   fprintf(stream,"DOUBLE\n");
   fprintf(stream,"%d\n",elements);
   for (i = 0; i < elements; i++) {

   }  


*/


static void gen_data_fread_ascii_header(gen_data_type * gen_data , FILE * stream) {
  util_fskip_lines(stream , 1); /* We know the first line contains "ASCII". */
  {
    int kw_length;
    fscanf(stream, "%d:",&kw_length);
    gen_data->ext_tag = util_fscanf_alloc_token(stream);
    if (kw_length > 0) {
      if (strlen(gen_data->ext_tag) != kw_length)
	util_abort("%s: something wrong with header:%d:%s \n",__func__ , kw_length , gen_data->ext_tag);
    } else {
      if (gen_data->ext_tag != NULL)
	util_abort("%s: something wrong with header:%d:%s \n",__func__ , kw_length , gen_data->ext_tag);
    }
  }
  
  {
    char * string_type;
    string_type = util_fscanf_alloc_token(stream);
    util_strupr(string_type);
    if (strcmp(string_type, "DOUBLE") == 0)
      gen_data->ecl_type = ecl_double_type;
    else if (strcmp(string_type, "FLOAT") == 0)
      gen_data->ecl_type = ecl_float_type;
    else 
      util_abort("%s: type identiefier:%s  not recognized - valid values are FLOAT | DOUBLE \n",__func__ , string_type);
    free(string_type);
  }
  
  if (!util_fscanf_int(stream , &gen_data->size))
    util_abort("%s: Failed to read the number of elements when parsing:%s. \n",__func__ , gen_data->src_file);
}


static void gen_data_fread_binary_C_header(gen_data_type * gen_data , FILE * stream) {
  util_exit("%s: not implemented yet ... \n");
}


static void gen_data_fread_binary_fortran_header(gen_data_type * gen_data , FILE * stream) {
  util_exit("%s: not implemented yet ... \n");
}


static void gen_data_fread_header(gen_data_type * gen_data , FILE * stream) {
  switch (gen_data->file_type) {
  case(ascii_file):
    gen_data_fread_ascii_header(gen_data , stream);
    break;
  case(binary_C_file):
    gen_data_fread_binary_C_header(gen_data , stream);
    break;
  case(binary_fortran_file):
    gen_data_fread_binary_fortran_header(gen_data , stream);
    break;
  default:
    util_abort("%s: internal error - invalid value in switch statement. \n",__func__);
  }
}


static void gen_data_fread_binary_C_data(gen_data_type * gen_data , FILE * stream) {
  util_exit("%s: not implemented yet ... \n");
}


static void gen_data_fread_binary_fortran_data(gen_data_type * gen_data , FILE * stream) {
  util_exit("%s: not implemented yet ... \n");
}


static void gen_data_fread_ascii_data(gen_data_type * gen_data , FILE *stream) {
  int i;
  switch (gen_data->ecl_type) {
  case(ecl_float_type):
    {
      float * data = (float *) gen_data->data;
      for (i=0; i < gen_data->size; i++)
	if (fscanf(stream,"%g",&data[i]) != 1)
	  util_abort("%s: failed to read element %d/%d from %s. \n",__func__ , (i+1), gen_data->size , gen_data->src_file);
    }
    break;
  case(ecl_double_type):
    {
      double * data = (double *) gen_data->data;
      for (i=0; i < gen_data->size; i++)
	if (fscanf(stream,"%lg",&data[i]) != 1)
	  util_abort("%s: failed to read element %d/%d from %s. \n",__func__ , (i+1), gen_data->size , gen_data->src_file);
    }
    break;
  default:
    util_abort("%s: unrecognized/not supported data_type:%d.\n",__func__ , gen_data->ecl_type);
  }
}


static void gen_data_fread_data(gen_data_type * gen_data , FILE * stream) {
  switch (gen_data->file_type) {
  case(ascii_file):
    gen_data_fread_ascii_data(gen_data , stream);
    break;
  case(binary_C_file):
    gen_data_fread_binary_C_data(gen_data , stream);
    break;
  case(binary_fortran_file):
    gen_data_fread_binary_fortran_data(gen_data , stream);
    break;
  default:
    util_abort("%s: internal error - invalid value in switch statement. \n",__func__);
  }
}


static void gen_data_set_file_data(gen_data_type * gen_data , const char * filename ) {
  char buffer[32];
  gen_data->src_file = util_alloc_string_copy(filename);
  if (fortio_is_fortran_file(filename , &gen_data->fortran_endian_flip)) {
    int record_length;
    fortio_type * fortio = fortio_fopen(filename , "r" , gen_data->fortran_endian_flip);
    record_length = fortio_fread_record(fortio , buffer);
    if (record_length != 6) 
      util_abort("%s: could not locate \'BINARY\' header in %s.\n",__func__ , filename);
    buffer[6] = '\0';
    util_strupr(buffer);
    if (strcmp(buffer , "BINARY") != 0)
      util_abort("%s: could not locate \'BINARY\' header in %s.\n",__func__ , filename);
    
    fortio_fclose(fortio);
    gen_data->file_type = binary_fortran_file;
  } else {
    FILE * stream = util_fopen(filename , "r");
    long int init_pos = ftell(stream);
    util_fread(buffer , 1 , 5 , stream , __func__);
    buffer[5] = '\0';
    util_strupr(buffer);
    if (strcmp(buffer , "ASCII") == 0)
      gen_data->file_type = ascii_file;
    else {
      fseek(stream , init_pos , SEEK_SET);
      util_fread(buffer , 1 , 6 , stream , __func__);
      buffer[6] = '\0';
      util_strupr(buffer);
      if (strcmp(buffer , "BINARY") == 0) 
	gen_data->file_type = binary_C_file;
      else 
	util_abort("%s: could not determine BINARY / ASCII status of file:%s. Header: %s not recognized \n",__func__ , filename , buffer);
    }
    fclose(stream);
  }
}

static void gen_data_alloc_empty(gen_data_config_type * config) {
  gen_data_type * gen_data = util_malloc(sizeof * gen_data, __func__);
  gen_data->config   = config;
  gen_data->data     = NULL;
  gen_data->ext_tag  = NULL;
  gen_data->src_file = NULL;
  gen_data->size     = 0;
  gen_data->active   = false;
}
  
  


void gen_data_free(gen_data_type * gen_data) {
  util_safe_free(gen_data->data);
  util_safe_free(gen_data->src_file);
  util_safe_free(gen_data->ext_tag);
  free(gen_data);
}


void gen_data_realloc_data(gen_data_type * gen_data) {
  gen_data->data = util_realloc(gen_data->data , gen_data->size * ecl_util_get_sizeof_ctype(gen_data->ecl_type) , __func__);
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
    util_fwrite_compressed(gen_data->data , gen_data->size * ecl_util_get_sizeof_ctype(gen_data->ecl_type) , stream);
  }
}


void gen_data_fread(gen_data_type * gen_data , FILE * stream) {
  DEBUG_ASSERT(gen_data)
  enkf_util_fread_assert_target_type(stream , GEN_DATA , __func__);
  gen_data->active = util_fread_bool(stream);
  if (gen_data->active) {
    gen_data->size = util_fread_int(stream);
    gen_data->size = util_fread_int(stream);
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
