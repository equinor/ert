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
#include <gen_data_common.h>
#include <gen_common.h>
#include <matrix.h>
#include <math.h>
#include <log.h>

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
  gen_data_config_type  * config;               /* Thin config object - mainly contains filename for remote load */
  char                  * data;                 /* Actual storage - will be casted to double or float on use. */
  int                     current_report_step;  /* Need this to look up the correct size in the config object. */
};




gen_data_config_type * gen_data_get_config(const gen_data_type * gen_data) { return gen_data->config; }


void gen_data_realloc_data(gen_data_type * gen_data) {
  int byte_size = gen_data_get_current_byte_size(gen_data);
  
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
  gen_data->__type_id         = GEN_DATA;
  gen_data->current_data_size = -1;  /* God - if you ever read this .... */
  return gen_data;
}


gen_data_type * gen_data_copyc(const gen_data_type * gen_data) {
  gen_data_type * copy = gen_data_alloc(gen_data->config);
  
  if (gen_data->data != NULL) {
    int byte_size = gen_data_get_current_byte_size( gen_data );
    copy->data    = util_alloc_copy(gen_data->data , byte_size , __func__);
  }
  copy->current_data_size = gen_data->current_data_size;
  return copy;
}
  
  
void gen_data_free_data(gen_data_type * gen_data) {
  util_safe_free(gen_data->data);
  gen_data->data              = NULL;
  gen_data->current_data_size = -1;
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


bool gen_data_store(const gen_data_type * gen_data , buffer_type * buffer , int report_step , bool internal_state) {
  const bool write_zero_size = true; /* true:ALWAYS write a file   false:only write files with size > 0. */
  {
    bool write    = write_zero_size;
    int size      = gen_data->current_data_size;
    if (size > 0) 
      write = true;
    
    if (write) {
      int byte_size = gen_data_get_current_byte_size( gen_data );
      buffer_fwrite_int( buffer , GEN_DATA );
      buffer_fwrite_int( buffer , size );
      buffer_fwrite_int( buffer , report_step);   /* Why the heck do I need to store this ???? */
      
      buffer_fwrite_compressed( buffer , gen_data->data , byte_size);
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


void gen_data_load(gen_data_type * gen_data , buffer_type * buffer) {
  int size;
  int report_step;

  enkf_util_assert_buffer_type(buffer , GEN_DATA);
  size           = buffer_fread_int(buffer);
  report_step    = buffer_fread_int(buffer);
  {
    size_t byte_size       = size * ecl_util_get_sizeof_ctype( gen_data_config_get_internal_type ( gen_data->config ));
    size_t compressed_size = buffer_get_remaining_size( buffer ); 
    gen_data->data         = util_realloc( gen_data->data , byte_size , __func__);
    buffer_fread_compressed( buffer , compressed_size , gen_data->data , byte_size );
  }
  gen_data_config_assert_size(gen_data->config , size , report_step);
  gen_data->current_data_size = size;
}




void gen_data_upgrade_103(const char * filename) {
  FILE * stream               = util_fopen(filename , "r");
  enkf_impl_type impl_type    = util_fread_int( stream );
  int 		 size         = util_fread_int( stream );
  int 		 report_step  = util_fread_int( stream );
  size_t byte_size            = util_fread_sizeof_compressed( stream );
  void		  * data      = util_fread_alloc_compressed( stream );
  fclose(stream);
  {
    buffer_type * buffer = buffer_alloc( 100 );
    buffer_fwrite_time_t( buffer , time(NULL));
    buffer_fwrite_int( buffer , impl_type );
    buffer_fwrite_int( buffer , size );
    buffer_fwrite_int( buffer , report_step );
    buffer_fwrite_compressed( buffer , data , byte_size);
    
    buffer_store( buffer , filename );
    buffer_free( buffer );
  }
}




int gen_data_serialize(const gen_data_type *gen_data ,serial_state_type * serial_state , size_t serial_offset , serial_vector_type * serial_vector) {
  ecl_type_enum ecl_type 	       = gen_data_config_get_internal_type(gen_data->config);
  const int data_size    	       = gen_data->current_data_size;
  const active_list_type  *active_list = gen_data_config_get_active_list(gen_data->config);
  int elements_added;

  elements_added = enkf_serialize(gen_data->data , data_size , ecl_type , active_list , serial_state ,serial_offset , serial_vector);
  return elements_added;
}



void gen_data_deserialize(gen_data_type * gen_data , serial_state_type * serial_state , const serial_vector_type * serial_vector) {
  ecl_type_enum ecl_type              = gen_data_config_get_internal_type(gen_data->config);
  const int data_size    	      = gen_data->current_data_size;
  const active_list_type *active_list = gen_data_config_get_active_list(gen_data->config);
  
  enkf_deserialize(gen_data->data , data_size , ecl_type , active_list  , serial_state , serial_vector);
}



void gen_data_matrix_serialize(const gen_data_type * gen_data , const active_list_type * active_list , matrix_type * A , int row_offset , int column) {
  const gen_data_config_type *config   = gen_data->config;
  const int                data_size   = gen_data->current_data_size;
  ecl_type_enum ecl_type               = gen_data_config_get_internal_type( config );

  enkf_matrix_serialize( gen_data->data , data_size , ecl_type , active_list , A , row_offset , column);
}


void gen_data_matrix_deserialize(gen_data_type * gen_data , const active_list_type * active_list , const matrix_type * A , int row_offset , int column) {
  const gen_data_config_type *config   = gen_data->config;
  const int                data_size   = gen_data->current_data_size;
  ecl_type_enum ecl_type               = gen_data_config_get_internal_type(config);
  
  enkf_matrix_deserialize( gen_data->data , data_size , ecl_type , active_list , A , row_offset , column);
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
      memcpy(gen_data->data , data , gen_data_get_current_byte_size( gen_data ));
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

   Return value is whether file was found - might have to check this
   in calling scope.
*/

bool gen_data_fload( gen_data_type * gen_data , const char * filename , int report_step) {
  bool   has_file = util_file_exists(filename);
  void * buffer   = NULL;
  int    size     = 0;
  ecl_type_enum load_type;
  
  if ( has_file ) {
    ecl_type_enum internal_type            = gen_data_config_get_internal_type(gen_data->config);
    gen_data_file_format_type input_format = gen_data_config_get_input_format( gen_data->config );
    buffer = gen_common_fload_alloc( filename , input_format , internal_type , &load_type , &size);
  } 
  gen_data->current_data_size = size;
  gen_data_set_data__(gen_data , size , report_step , load_type , buffer);
  util_safe_free(buffer);
  
  return has_file;
}




/**
   The return value from the xxx_ecl_load() functions should be
   true|false whether the load has worked out OK. The
   summary_ecl_load() and field_ecl_load() functions should only be
   called when we insist that there should be data waiting, and
   consequently a false return value should be interpreted as a failure. 
   
   For the gen_data_ecl_load() function we do not know if we can
   expect to find data (that should probably be internalized in the
   config object ... ), hence the function must return true anyway, to
   not signal false errors.
*/



bool gen_data_ecl_load(gen_data_type * gen_data , const char * ecl_file , const ecl_sum_type * ecl_sum, const ecl_file_type * restart_file , int report_step) {
  gen_data_fload( gen_data , ecl_file , report_step );
  return true;
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
    if (!gen_data_fload(gen_data , init_file , 0))
      util_abort("%s: could not find file:%s \n",__func__ , init_file);
    free(init_file);
    return true;
  } else
    return false; /* No init performed ... */
}





static void gen_data_ecl_write_ASCII(const gen_data_type * gen_data , const char * file , gen_data_file_format_type export_format) {
  FILE * stream   = util_fopen(file , "w");
  char * template_buffer;
  int    template_data_offset, template_buffer_size , template_data_skip;

  if (export_format == ASCII_TEMPLATE) {
    gen_data_config_get_template_data( gen_data->config , &template_buffer , &template_data_offset , &template_buffer_size , &template_data_skip);
    util_fwrite( template_buffer , 1 , template_data_offset , stream , __func__);
  }
  
  {
    ecl_type_enum internal_type = gen_data_config_get_internal_type(gen_data->config);
    const int size              = gen_data->current_data_size;
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
  
  if (export_format == ASCII_TEMPLATE) {
    int new_offset = template_data_offset + template_data_skip;
    util_fwrite( &template_buffer[new_offset] , 1 , template_buffer_size - new_offset , stream , __func__);
  }
  fclose(stream);
}



static void gen_data_ecl_write_binary(const gen_data_type * gen_data , const char * file , ecl_type_enum export_type) {
  FILE * stream    = util_fopen(file , "w");
  int sizeof_ctype = ecl_util_get_sizeof_ctype( export_type );
  util_fwrite( gen_data->data , sizeof_ctype , gen_data->current_data_size , stream , __func__);
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
    case(ASCII_TEMPLATE):
      gen_data_ecl_write_ASCII(gen_data , full_path , export_type);
      break;
    case(BINARY_DOUBLE):
      gen_data_ecl_write_binary(gen_data , full_path , ecl_double_type);
      break;
    case(BINARY_FLOAT):
      gen_data_ecl_write_binary(gen_data , full_path , ecl_float_type);
      break;
    default:
      util_abort("%s: internal error - export type is not set.\n",__func__);
    }

    free( full_path );
  }
}


static void gen_data_assert_index(const gen_data_type * gen_data, int index) {
  if ((index < 0) || (index >= gen_data->current_data_size))
    util_abort("%s: index:%d invalid. Valid range: [0,%d) \n",__func__ , index , gen_data->current_data_size);
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

  if (index_key != NULL) {
    if (util_sscanf_int(index_key , &index)) {
      if (index < gen_data->current_data_size) {
        *valid = true;
        return gen_data_iget_double( gen_data , index );
      }
    } 
  }
  
  *valid = false;
  return -1; /* Dummy to shut up compiler */
}


int gen_data_get_size(const gen_data_type * gen_data) {
  return gen_data->current_data_size;
}


const char * gen_data_get_key( const gen_data_type * gen_data) {
  return gen_data_config_get_key( gen_data->config );
}


void gen_data_clear( gen_data_type * gen_data ) {
  const gen_data_config_type * config = gen_data->config;
  ecl_type_enum internal_type         = gen_data_config_get_internal_type( config );
  const int data_size                 = gen_data->current_data_size;

  if (internal_type == ecl_float_type) {
    float * data = (float * ) gen_data->data;
    for (int i = 0; i < data_size; i++)
      data[i] = 0;
  } else if (internal_type == ecl_double_type) {
    double * data = (double * ) gen_data->data;
    for (int i = 0; i < data_size; i++)
      data[i] = 0;
  } 
}



void gen_data_isqrt(gen_data_type * gen_data) {
  const int data_size               = gen_data->current_data_size;
  const ecl_type_enum internal_type = gen_data_config_get_internal_type(gen_data->config);
  
  if (internal_type == ecl_float_type) {
    float * data = (float *) gen_data->data;
    for (int i=0; i < data_size; i++)
      data[i] = sqrtf( data[i] );
  } else if (internal_type == ecl_double_type) {
    double * data = (double *) gen_data->data;
    for (int i=0; i < data_size; i++)
      data[i] = sqrt( data[i] );
  }
}




void gen_data_iadd(gen_data_type * gen_data1, const gen_data_type * gen_data2) {
  //gen_data_config_assert_binary(gen_data1->config , gen_data2->config , __func__); 
  {
    const int data_size               = gen_data1->current_data_size;
    const ecl_type_enum internal_type = gen_data_config_get_internal_type(gen_data1->config);
    int i;

    if (internal_type == ecl_float_type) {
      float * data1       = (float *) gen_data1->data;
      const float * data2 = (const float *) gen_data2->data;
      for (i = 0; i < data_size; i++)
	data1[i] += data2[i];
    } else if (internal_type == ecl_double_type) {
      double * data1       = (double *) gen_data1->data;
      const double * data2 = (const double *) gen_data2->data;
      for (i = 0; i < data_size; i++) {
        printf("%s:%d\n",__func__ , i);
	data1[i] += data2[i];
      }
    }
  }
}


void gen_data_imul(gen_data_type * gen_data1, const gen_data_type * gen_data2) {
  //gen_data_config_assert_binary(gen_data1->config , gen_data2->config , __func__); 
  {
    const int data_size               = gen_data1->current_data_size;
    const ecl_type_enum internal_type = gen_data_config_get_internal_type(gen_data1->config);
    int i;

    if (internal_type == ecl_float_type) {
      float * data1       = (float *) gen_data1->data;
      const float * data2 = (const float *) gen_data2->data;
      for (i = 0; i < data_size; i++)
	data1[i] *= data2[i];
    } else if (internal_type == ecl_double_type) {
      double * data1       = (double *) gen_data1->data;
      const double * data2 = (const double *) gen_data2->data;
      for (i = 0; i < data_size; i++)
	data1[i] *= data2[i];
    }
  }
}


void gen_data_iaddsqr(gen_data_type * gen_data1, const gen_data_type * gen_data2) {
  //gen_data_config_assert_binary(gen_data1->config , gen_data2->config , __func__); 
  {
    const int data_size               = gen_data1->current_data_size;
    const ecl_type_enum internal_type = gen_data_config_get_internal_type(gen_data1->config);
    int i;

    if (internal_type == ecl_float_type) {
      float * data1       = (float *) gen_data1->data;
      const float * data2 = (const float *) gen_data2->data;
      for (i = 0; i < data_size; i++)
	data1[i] += data2[i] * data2[i];
    } else if (internal_type == ecl_double_type) {
      double * data1       = (double *) gen_data1->data;
      const double * data2 = (const double *) gen_data2->data;
      for (i = 0; i < data_size; i++)
	data1[i] += data2[i] * data2[i];
    }
  }
}


void gen_data_scale(gen_data_type * gen_data, double scale_factor) {
  //gen_data_config_assert_unary(gen_data->config, __func__); 
  {
    const int data_size               = gen_data->current_data_size;
    const ecl_type_enum internal_type = gen_data_config_get_internal_type(gen_data->config);
    int i;

    if (internal_type == ecl_float_type) {
      float * data       = (float *) gen_data->data;
      for (i = 0; i < data_size; i++)
	data[i] *= scale_factor;
    } else if (internal_type == ecl_double_type) {
      double * data       = (double *) gen_data->data;
      for (i = 0; i < data_size; i++)
	data[i] *= scale_factor;
    }
  }
}




#define INFLATE(inf,std,min,logh)                                                                                                                                \
{                                                                                                                                                                \
   for (int i=0; i < data_size; i++) {                                                                                                                           \
     if (std_data[i] > 0)                                                                                                                                        \
        inflation_data[i] = util_float_max( 1.0 , min_std_data[i] / std_data[i]);                                                                                \
      else                                                                                                                                                       \
        inflation_data[i] = 1.0;                                                                                                                                 \
   }                                                                                                                                                             \
   if (add_log_entry) {                                                                                                                                          \
     for (int c=0; c < data_size; c++) {                                                                                                                         \
       if (inflation_data[c] > 1.0)                                                                                                                              \
         log_add_fmt_message( logh , log_level , NULL , "Inflating %s:%d with %6.4f" , gen_data_config_get_key( inflation->config ) , c, inflation_data[c]);     \
     }                                                                                                                                                           \
   }                                                                                                                                                             \
}                                                                   


/**
   If the size changes during the simulation this will go 100% belly up.
*/

void gen_data_set_inflation(gen_data_type * inflation , const gen_data_type * std , const gen_data_type * min_std , log_type * logh) {
  const int log_level              = 3;
  const gen_data_config_type * config = inflation->config;
  ecl_type_enum ecl_type              = gen_data_config_get_internal_type( config );
  const int data_size                 = std->current_data_size;
  bool add_log_entry = false;
  if (log_get_level( logh ) >= log_level)
    add_log_entry = true;


  if (ecl_type == ecl_float_type) {
    float       * inflation_data = (float *)       inflation->data;
    const float * std_data       = (const float *) std->data;
    const float * min_std_data   = (const float *) min_std->data;
    
    INFLATE(inflation_data , std_data , min_std_data , logh);
    
  } else {
    double       * inflation_data = (double *)       inflation->data;
    const double * std_data       = (const double *) std->data;
    const double * min_std_data   = (const double *) min_std->data;
    
    INFLATE(inflation_data , std_data , min_std_data , logh);
  }
}
#undef INFLATE


/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/
SAFE_CONST_CAST(gen_data , GEN_DATA)
SAFE_CAST(gen_data , GEN_DATA)
VOID_USER_GET(gen_data)
VOID_ALLOC(gen_data)
VOID_FREE(gen_data)
VOID_FREE_DATA(gen_data)
VOID_REALLOC_DATA(gen_data)
VOID_COPYC     (gen_data)
VOID_SERIALIZE(gen_data)
VOID_DESERIALIZE(gen_data)
VOID_INITIALIZE(gen_data)
VOID_ECL_WRITE(gen_data)
VOID_ECL_LOAD(gen_data)
VOID_LOAD(gen_data);
VOID_STORE(gen_data);
VOID_MATRIX_SERIALIZE(gen_data)
VOID_MATRIX_DESERIALIZE(gen_data)
VOID_SET_INFLATION(gen_data)
VOID_CLEAR(gen_data)
VOID_SCALE(gen_data)
VOID_IMUL(gen_data)
VOID_IADD(gen_data)
VOID_IADDSQR(gen_data)
VOID_ISQRT(gen_data)
