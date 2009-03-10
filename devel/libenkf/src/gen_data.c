#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <util.h>
#include <ecl_sum.h>
#include <ecl_file.h>
#include <fortio.h>
#include <ecl_util.h>
#include <enkf_serialize.h>
#include <enkf_types.h>
#include <enkf_macros.h>
#include <enkf_util.h>
#include <gen_data_config.h>
#include <gen_data.h>
#include <gen_common.h>



/**
   The file implements a general data type which can be used to update
   arbitrary data which the EnKF system has *ABSOLUTELY NO IDEA* of
   how is organised; how it should be used in the forward model and so
   on. Similarly to the field objects, the gen_data objects can be
   treated both as parameters and as dynamic data.

   Whether the ecl_load function should be called (i.e. it is dynamic
   data) is determined at the enkf_node level, and no busissiness of
   the gen_data implementation.
*/
   



struct gen_data_struct {
  int                     __type_id;
  gen_data_config_type  * config;    /* Thin config object - mainly contains filename for remote load */
  char                  * data;     /* Actual storage - will be casted to double or float on use. */
};




gen_data_config_type * gen_data_get_config(const gen_data_type * gen_data) { return gen_data->config; }


void gen_data_realloc_data(gen_data_type * gen_data) {
  int byte_size = gen_data_config_get_byte_size(gen_data->config);

  if (byte_size > 0)
    gen_data->data = util_realloc(gen_data->data , byte_size , __func__);
  else 
    gen_data->data = util_safe_free( gen_data->data );
    
}



gen_data_type * gen_data_alloc(const gen_data_config_type * config) {
  gen_data_type * gen_data = util_malloc(sizeof * gen_data, __func__);
  gen_data->config    = (gen_data_config_type *) config;
  gen_data->data      = NULL;
  gen_data_realloc_data(gen_data);
  gen_data->__type_id = GEN_DATA;
  return gen_data;
}


gen_data_type * gen_data_copyc(const gen_data_type * gen_data) {
  gen_data_type * copy = gen_data_alloc(gen_data->config);
  
  if (gen_data->data != NULL) {
    int byte_size = gen_data_config_get_byte_size( gen_data->config );
    copy->data = util_alloc_copy(gen_data->data , byte_size , __func__);
  }
  
  return copy;
}
  
  
void gen_data_free_data(gen_data_type * gen_data) {
  util_safe_free(gen_data->data);
  gen_data->data = NULL;
}



void gen_data_free(gen_data_type * gen_data) {
  gen_data_free_data(gen_data);
  free(gen_data);
}




/**
   Observe that this function writes parameter size to disk, that is
   special. The reason is that the config object does not know the
   size (on allocation).

   The function currently writes an empty file (with only a report
   step and a size == 0) in the case where it does not have data. This
   is controlled by the value of the variable write_zero_size; if this
   is changed to false some semantics in the laod code must be
   changed.
*/

bool gen_data_fwrite(const gen_data_type * gen_data , FILE * stream , bool internal_state) {
  const bool write_zero_size = true; /* true:ALWAYS write a file   false:only write files with size > 0. */
  {
    bool write      = write_zero_size;
    int size        = gen_data_config_get_data_size(gen_data->config);
    int report_step = gen_data_config_get_report_step(gen_data->config); 
    if (size > 0) 
      write = true;

    if (write) {
      int byte_size = gen_data_config_get_byte_size(gen_data->config);
      
      enkf_util_fwrite_target_type(stream , GEN_DATA);
      util_fwrite_int(size        , stream);
      util_fwrite_int(report_step , stream);
      util_fwrite_compressed(gen_data->data , byte_size , stream);
      return true;
    } else
      return false;   /* When false is returned - the (empty) file will be removed */
  }
}


/* 
   Observe that this function manipulates memory directly. This should
   ideally be left to the enkf_node layer, but for this type the data
   size is determined at load time.
*/

void gen_data_fread(gen_data_type * gen_data , FILE * stream) {
  int size;
  int report_step;
  enkf_util_fread_assert_target_type(stream , GEN_DATA);
  size        = util_fread_int(stream);
  report_step = util_fread_int(stream);
  util_safe_free(gen_data->data);
  gen_data->data = util_fread_alloc_compressed(stream);
  gen_data_config_assert_size(gen_data->config , size , report_step);
}




int gen_data_serialize(const gen_data_type *gen_data ,serial_state_type * serial_state , size_t serial_offset , serial_vector_type * serial_vector) {
  ecl_type_enum ecl_type 	       = gen_data_config_get_internal_type(gen_data->config);
  const int data_size    	       = gen_data_config_get_data_size(gen_data->config);
  const active_list_type  *active_list = gen_data_config_get_active_list(gen_data->config);
  int elements_added;

  elements_added = enkf_serialize(gen_data->data , data_size , ecl_type , active_list , serial_state ,serial_offset , serial_vector);
  return elements_added;
}



void gen_data_deserialize(gen_data_type * gen_data , serial_state_type * serial_state , const serial_vector_type * serial_vector) {
  ecl_type_enum ecl_type              = gen_data_config_get_internal_type(gen_data->config);
  const int data_size    	      = gen_data_config_get_data_size(gen_data->config);
  const active_list_type *active_list = gen_data_config_get_active_list(gen_data->config);
  
  enkf_deserialize(gen_data->data , data_size , ecl_type , active_list  , serial_state , serial_vector);
}



/*
  This function sets the data field of the gen_data instance after the
  data has been loaded from file.
*/ 
static void gen_data_set_data__(gen_data_type * gen_data , int size, int report_step , ecl_type_enum load_type , const void * data) {
  gen_data_config_assert_size(gen_data->config , size, report_step);
  gen_data_realloc_data(gen_data);
  if (size > 0) {
    ecl_type_enum internal_type = gen_data_config_get_internal_type(gen_data->config);

    if (load_type == internal_type)
      memcpy(gen_data->data , data , gen_data_config_get_byte_size(gen_data->config));
    else {
      if (load_type == ecl_float_type)
	util_float_to_double((double *) gen_data->data , data , size);
      else
	util_double_to_float((float *) gen_data->data , data , size);
    }
  }
}
      
      



/**
   This functions loads data from file. Observe that there is *NO*
   header information in this file - the size is determined by seeing
   how much can be successfully loaded.

   The file is loaded with the gen_common_fload_alloc() function, and
   can be in formatted ASCII or binary_float / binary_double. 

   When the read is complete it is checked/verified with the config
   object that this file was as long as the others we have loaded for
   other members; it is perfectly OK for the file to not exist. In
   which case a size of zero is set, for this report step.
*/

void gen_data_ecl_load(gen_data_type * gen_data , const char * ecl_file , const ecl_sum_type * ecl_sum, const ecl_file_type * restart_file , int report_step) {
  {
    void * buffer = NULL;
    int    size   = 0;
    ecl_type_enum load_type;
    
    if (util_file_exists(ecl_file)) {
      ecl_type_enum internal_type            = gen_data_config_get_internal_type(gen_data->config);
      gen_data_file_format_type input_format = gen_data_config_get_input_format( gen_data->config );
      buffer = gen_common_fload_alloc( ecl_file , input_format , internal_type , &load_type , &size);
    } 
    
    gen_data_set_data__(gen_data , size , report_step , load_type , buffer);
    util_safe_free(buffer);
  }
}


    
/**
   This function initializes the parameter. This is based on loading a
   file. The name of the file is derived from a path_fmt instance
   owned by the config object. Observe that there is *NO* header
   information in this file. We just read floating point numbers until
   we reach EOF.
   
   When the read is complete it is checked/verified with the config
   object that this file was as long as the files we have loaded for
   other members.
   
   If gen_data_config_alloc_initfile() returns NULL that means that
   the gen_data instance does not have any init function - that is OK.
*/



bool gen_data_initialize(gen_data_type * gen_data , int iens) {
  char * init_file = gen_data_config_alloc_initfile(gen_data->config , iens);
  if (init_file != NULL) {
    gen_data_ecl_load(gen_data , init_file , NULL , NULL , 0);
    free(init_file);
    return true;
  } else
    return false; /* No init performed ... */
}





static void gen_data_ecl_write_ASCII(const gen_data_type * gen_data , const char * file , gen_data_file_format_type export_format) {
  FILE * stream   = util_fopen(file , "w");
  char * template_buffer;
  int    template_data_offset, template_buffer_size , template_data_skip;

  if (export_format == ASCII_template) {
    gen_data_config_get_template_data( gen_data->config , &template_buffer , &template_data_offset , &template_buffer_size , &template_data_skip);
    util_fwrite( template_buffer , 1 , template_data_offset , stream , __func__);
  }
  
  {
    ecl_type_enum internal_type = gen_data_config_get_internal_type(gen_data->config);
    const int size              = gen_data_config_get_data_size(gen_data->config); 
    int i;
    if (internal_type == ecl_float_type) {
      float * float_data = (float *) gen_data->data;
      for (i=0; i < size; i++)
	fprintf(stream , "%g\n",float_data[i]);
    } else if (internal_type == ecl_double_type) {
      double * double_data = (double *) gen_data->data;
      for (i=0; i < size; i++)
	fprintf(stream , "%lg\n",double_data[i]);
    } else 
      util_abort("%s: internal error - wrong type \n",__func__);
  }
  
  if (export_format == ASCII_template) {
    int new_offset = template_data_offset + template_data_skip;
    util_fwrite( &template_buffer[new_offset] , 1 , template_buffer_size - new_offset , stream , __func__);
  }
  fclose(stream);
}



static void gen_data_ecl_write_binary(const gen_data_type * gen_data , const char * file , ecl_type_enum export_type) {
  FILE * stream    = util_fopen(file , "w");
  int sizeof_ctype = ecl_util_get_sizeof_ctype( export_type );
  util_fwrite( gen_data->data , sizeof_ctype , gen_data_config_get_data_size( gen_data->config ) , stream , __func__);
  fclose(stream);
}



/** 
    It is the enkf_node layer which knows whether the node actually
    has any data to export. If it is not supposed to write data to the
    forward model, i.e. it is of enkf_type 'dynamic_result' that is
    signaled down here with eclfile == NULL.
*/


void gen_data_ecl_write(const gen_data_type * gen_data , const char * run_path , const char * eclfile , fortio_type * fortio) {
  if (eclfile != NULL) {  
    char * full_path = util_alloc_filename( run_path , eclfile  , NULL);

    gen_data_file_format_type export_type = gen_data_config_get_output_format( gen_data->config );
    switch (export_type) {
    case(ASCII):
      gen_data_ecl_write_ASCII(gen_data , full_path , export_type);
      break;
    case(ASCII_template):
      gen_data_ecl_write_ASCII(gen_data , full_path , export_type);
      break;
    case(binary_double):
      gen_data_ecl_write_binary(gen_data , full_path , ecl_double_type);
      break;
    case(binary_float):
      gen_data_ecl_write_binary(gen_data , full_path , ecl_float_type);
      break;
    default:
      util_abort("%s: internal error \n",__func__);
    }

    free( full_path );
  }
}


static void gen_data_assert_index(const gen_data_type * gen_data, int index) {
  if ((index < 0) || (index >= gen_data_config_get_data_size(gen_data->config)))
    util_abort("%s: index:%d invalid. Valid range: [0,%d) \n",__func__ , index , gen_data_config_get_data_size(gen_data->config));
}


double gen_data_iget_double(const gen_data_type * gen_data, int index) {
  gen_data_assert_index(gen_data , index); 
  {
    ecl_type_enum internal_type = gen_data_config_get_internal_type(gen_data->config);
    if (internal_type == ecl_double_type) {
      double * data = (double *) gen_data->data;
      return data[index];
    } else {
      float * data = (float *) gen_data->data;
      return data[index];
    }
  }
}



/**
   The filesystem will (currently) store gen_data instances which do
   not hold any data. Therefor it will be quite common to enter this
   function with an empty instance, we therefor just set valid =>
   false, and return silently in that case.
*/

double gen_data_user_get(const gen_data_type * gen_data, const char * index_key, bool * valid)
{
  int index;

  if (util_sscanf_int(index_key , &index)) {
    if (index < gen_data_config_get_data_size(gen_data->config)) {
      *valid = true;
      return gen_data_iget_double( gen_data , index );
    }
  } 

  *valid = false;
  return -1; /* Dummy to shut up compiler */
}


int gen_data_get_size(const gen_data_type * gen_data) {
  return gen_data_config_get_data_size( gen_data->config );
}



/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/

SAFE_CAST(gen_data , GEN_DATA)
VOID_USER_GET(gen_data)
VOID_ALLOC(gen_data)
VOID_FREE(gen_data)
VOID_FREE_DATA(gen_data)
VOID_REALLOC_DATA(gen_data)
VOID_FWRITE    (gen_data)
VOID_FREAD     (gen_data)
VOID_COPYC     (gen_data)
VOID_SERIALIZE(gen_data)
VOID_DESERIALIZE(gen_data)
VOID_INITIALIZE(gen_data)
VOID_ECL_WRITE(gen_data)
VOID_ECL_LOAD(gen_data)

