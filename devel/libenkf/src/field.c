#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <field.h>
#include <util.h>
#include <string.h>
#include <fortio.h>
#include <ecl_kw.h>
#include <ecl_fstate.h>
#include <field_config.h>
#include <rms_file.h>
#include <rms_tagkey.h>
#include <ecl_util.h>
#include <rms_type.h>
#include <rms_util.h>
#include <fortio.h>

#define  DEBUG
#define  TARGET_TYPE FIELD
#include "enkf_debug.h"



GET_DATA_SIZE_HEADER(field);

/*****************************************************************/

/**
  The field data type contains for "something" which is distributed
  over the full grid, i.e. permeability or pressure. All configuration
  information is stored in the config object, which is of type
  field_config_type. Observe the following:

  * The field **only** contains the active cells - the config object
    has a reference to actnum information.

  * The data is stored in a char pointer; the real underlying data can
    be (at least) of the types int, float and double.
*/

struct field_struct {
  DEBUG_DECLARE                                   /* A type identifier which can be used for run-time checks of casting operations. */
  const  field_config_type * config;              /* The field config object - containing information of active cells++ */
  char  *data;                                    /* The actual storage for the field - suitabley casted to int/float/double on use*/
             
  bool   shared_data;                             /* If the data is shared - i.e. managed (xalloc & free) from another scope. */
  int    shared_byte_size;                        /* The size of the shared buffer (if it is shared). */
};



#define EXPORT_MACRO                                                                           	  	  \
{                                                                                                 	  \
  int i,j,k;                                                                                      	  \
   for (k=0; k < config->nz; k++) {                                                               	  \
     for (j=0; j < config->ny; j++) {                                                             	  \
       for (i=0; i < config->nx; i++) {                                                           	  \
         int index1D = field_config_global_index(config , i , j , k);                             	  \
         int index3D;                                                                             	  \
         if (rms_index_order)                                               		     	       	  \
           index3D = rms_util_global_index_from_eclipse_ijk(config->nx,config->ny,config->nz,i,j,k);      \
         else                                                                       	       	          \
           index3D = i + j * config->nx + k* config->nx*config->ny;           	               	          \
         if (index1D >= 0)                                                                        	  \
   	   target_data[index3D] = src_data[index1D];                               	       	          \
         else                                                                                     	  \
           memcpy(&target_data[index3D] , fill_value , sizeof_ctype_target);                      	  \
        }                                                                                         	  \
     }                                                                                            	  \
   }                                                                                                      \
}                                                                                                         


void field_export3D(const field_type * field , void *_target_data , bool rms_index_order , ecl_type_enum target_type , void *fill_value) {
  const field_config_type * config = field->config;
  int   sizeof_ctype_target = ecl_util_get_sizeof_ctype(target_type);
  
  switch(config->ecl_type) {
  case(ecl_double_type):
    {
      const double * src_data = (const double *) field->data;
      if (target_type == ecl_float_type) {
	float *target_data = (float *) _target_data;
	EXPORT_MACRO;
      } else if (target_type == ecl_double_type) {
	double *target_data = (double *) _target_data;
	EXPORT_MACRO;
      } else {
	fprintf(stderr,"%s: double field can only export to double/float\n",__func__);
	abort();
      }
    }
    break;
  case(ecl_float_type):
    {
      const float * src_data = (const float *) field->data;
      if (target_type == ecl_float_type) {
	float *target_data = (float *) _target_data;
	EXPORT_MACRO;
      } else if (target_type == ecl_double_type) {
	double *target_data = (double *) _target_data;
	EXPORT_MACRO;
      } else {
	fprintf(stderr,"%s: float field can only export to double/float\n",__func__);
	abort();
      }
    }
    break;
  case(ecl_int_type):
    {
      const int * src_data = (const int *) field->data;
      if (target_type == ecl_float_type) {
	float *target_data = (float *) _target_data;
	EXPORT_MACRO;
      } else if (target_type == ecl_double_type) {
	double *target_data = (double *) _target_data;
	EXPORT_MACRO;
      } else if (target_type == ecl_int_type) {
	int *target_data = (int *) _target_data;
	EXPORT_MACRO;
      }  else {
	fprintf(stderr,"%s: int field can only export to int/double/float\n",__func__);
	abort();
      }
    }
    break;
  default:
    fprintf(stderr,"%s: Sorry field has unexportable type ... \n",__func__);
    break;
  }
}
#undef EXPORT_MACRO
  

/*****************************************************************/
#define IMPORT_MACRO                                                                           	            \
{                                                                                                           \
  int i,j,k;                                                                                                \
  for (k=0; k < config->nz; k++) {                                                               	    \
     for (j=0; j < config->ny; j++) {                                                             	    \
       for (i=0; i < config->nx; i++) {                                                           	    \
         int index1D = field_config_global_index(config , i , j , k);                             	    \
         int index3D;                                                                             	    \
         if (index1D >= 0) {                                                                      	    \
   	   if (rms_index_order)                                               		     	       	    \
   	     index3D = rms_util_global_index_from_eclipse_ijk(config->nx,config->ny,config->nz,i,j,k); 	    \
   	   else                                                                       	       	    	    \
   	     index3D = i + j * config->nx + k* config->nx*config->ny;           	               	    \
           target_data[index1D] = src_data[index3D] ;                               	       	    	    \
         }                                                                                         	    \
      }                                                                                           	    \
     }                                                                                            	    \
   }                                                                                                        \
}



/**
   The main function of the field_import3D and field_export3D
   functions are to skip the inactive cells (field_import3D) and
   distribute inactive cells (field_export3D). In addition we can
   reorganize input/output according to the RMS Roff index convention,
   and also perform float <-> double conversions.

   Observe that these functions only import/export onto memory
   buffers, the actual reading and writing of files is done in other
   functions (calling these).
*/

static void field_import3D(field_type * field , const void *_src_data , bool rms_index_order , ecl_type_enum src_type) {
  const field_config_type * config = field->config;
  
  switch(config->ecl_type) {
  case(ecl_double_type):
    {
      double * target_data = (double *) field->data;
      if (src_type == ecl_float_type) {
	float *src_data = (float *) _src_data;
	IMPORT_MACRO;
      } else if (src_type == ecl_double_type) {
	double *src_data = (double *) _src_data;
	IMPORT_MACRO;
      } else if (src_type == ecl_int_type) {
	int *src_data = (int *) _src_data;
	IMPORT_MACRO;
      } else {
	fprintf(stderr,"%s: double field can only import from int/double/float\n",__func__);
	abort();
      }
    }
    break;
  case(ecl_float_type):
    {
      float * target_data = (float *) field->data;
      if (src_type == ecl_float_type) {
	float *src_data = (float *) _src_data;
	IMPORT_MACRO;
      } else if (src_type == ecl_double_type) {
	double *src_data = (double *) _src_data;
	IMPORT_MACRO;
      } else if (src_type == ecl_int_type) {
	int *src_data = (int *) _src_data;
	IMPORT_MACRO;
      } else {
	fprintf(stderr,"%s: double field can only import from int/double/float\n",__func__);
	abort();
      }
    }
    break;
  case(ecl_int_type):
    {
      int * target_data = (int *) field->data;
      if (src_type == ecl_int_type) {
	int *src_data = (int *) _src_data;
	IMPORT_MACRO;
      }  else {
	fprintf(stderr,"%s: int field can only import from int\n",__func__);
	abort();
      }
    }
    break;
  default:
    fprintf(stderr,"%s: Sorry field has unimportable type ... \n",__func__);
    break;
  }
}
#undef IMPORT_MACRO


/*****************************************************************/

#define CLEAR_MACRO(d,s) { int k; for (k=0; k < (s); k++) (d)[k] = 0; }
void field_clear(field_type * field) {
  const ecl_type_enum ecl_type = field_config_get_ecl_type(field->config);
  const int data_size          = field_config_get_data_size(field->config);   

  switch (ecl_type) {
  case(ecl_double_type):
    {
      double * data = (double *) field->data;
      CLEAR_MACRO(data , data_size);
      break;
    }
  case(ecl_float_type):
    {
      float * data = (float *) field->data;
      CLEAR_MACRO(data , data_size);
      break;
    }
  case(ecl_int_type):
    {
      int * data = (int *) field->data;
      CLEAR_MACRO(data , data_size);
      break;
    }
  default:
    fprintf(stderr,"%s: not implemeneted for data_type: %d \n",__func__ , ecl_type);
  }
}
#undef CLEAR_MACRO


void field_realloc_data(field_type *field) {
  if (field->shared_data) {
    if (field_config_get_byte_size(field->config) > field->shared_byte_size) {
      fprintf(stderr,"%s: attempt to grow field with shared data - aborting \n",__func__);
      abort();
    }
  } else 
    field->data = util_malloc(field_config_get_byte_size(field->config) , __func__);
}



void field_free_data(field_type *field) {
  if (!field->shared_data) {
    free(field->data);
    field->data = NULL;
  }
}




static field_type * __field_alloc(const field_config_type * field_config , void * shared_data , int shared_byte_size) {
  field_type * field  = malloc(sizeof *field);
  field->config = field_config;
  if (shared_data == NULL) {
    field->data        = NULL;
    field->shared_data = false;
    field_realloc_data(field);
  } else {
    field->data             = shared_data;
    field->shared_data      = true;
    field->shared_byte_size = shared_byte_size;
    if (shared_byte_size < field_config_get_byte_size(field->config)) {
      fprintf(stderr,"%s: the shared buffer is to small to hold the input field - aborting \n",__func__);
      abort();
    }
  }
  DEBUG_ASSIGN(field)
  return field;
}


field_type * field_alloc(const field_config_type * field_config) {
  return __field_alloc(field_config , NULL , 0);
}


field_type * field_alloc_shared(const field_config_type * field_config, void * shared_data , int shared_byte_size) {
  return __field_alloc(field_config , shared_data , shared_byte_size);
}


field_type * field_copyc(const field_type *field) {
  field_type * new = field_alloc(field->config);
  memcpy(new->data , field->data , field_config_get_byte_size(field->config));
  return new;
}




void field_fread(field_type * field , FILE * stream) {
  int  data_size , sizeof_ctype;
  bool read_compressed;
  enkf_util_fread_assert_target_type(stream , FIELD );
  fread(&data_size     	  , sizeof  data_size        , 1 , stream);
  fread(&sizeof_ctype 	  , sizeof  sizeof_ctype     , 1 , stream);
  fread(&read_compressed  , sizeof  read_compressed  , 1 , stream);
  if (read_compressed)
    util_fread_compressed(field->data , stream);
  else
    enkf_util_fread(field->data , sizeof_ctype , data_size , stream , __func__);

}





static void * __field_alloc_3D_data(const field_type * field , int data_size , bool rms_index_order , ecl_type_enum ecl_type , ecl_type_enum target_type) {
  void * data = util_malloc(data_size * ecl_util_get_sizeof_ctype(target_type) , __func__);
  if (ecl_type == ecl_double_type) {
    double fill;
    if (rms_index_order)
      fill = RMS_INACTIVE_DOUBLE;
    else
      fill = 0;
    field_export3D(field , data , rms_index_order , target_type , &fill);
  } else if (ecl_type == ecl_float_type) {
    float fill;
    if (rms_index_order)
      fill = RMS_INACTIVE_FLOAT;
    else
      fill = 0;
    field_export3D(field , data , rms_index_order , target_type , &fill);
  } else if (ecl_type == ecl_int_type) {
    int fill;
    if (rms_index_order)
      fill = RMS_INACTIVE_INT;
    else
      fill = 0;
    field_export3D(field , data , rms_index_order , target_type , &fill);
  } else {
    fprintf(stderr,"%s: trying to export type != int/float/double - aborting \n",__func__);
    abort();
  }
  return data;
}


/**
   A general comment abourt writing fields to disk:

   The writing of fields to disk can be done in **MANY** different ways:

   o The native function field_fwrite() will save the field in the
     format most suitable for use with enkf. This function will only
     save the active cells, and compress the field if the variable
     write_compressed is true. Most of the configuration information
     is with the field_config object, and not saved with the field.

   o Export as ECLIPSE input. This again has three subdivisions:

     * The function field_ecl_grdecl_export() will write the field to
       disk in a format suitable for ECLIPSE INCLUDE statements. This
       means that both active and inactive cells are written, with a
       zero fill for the inactive.

     * The functions field_xxxx_fortio() writes the field in the
       ECLIPSE restart format. The function field_ecl_write3D_fortio()
       writes all the cells - with zero filling for inactive
       cells. This is suitable for IMPORT of e.g. PORO.
       
       The function field_ecl_write1D_fortio() will write only the
       active cells in an ECLIPSE restart file. This is suitable for
       e.g. the pressure.

       Observe that the function field_ecl_write() should get config
       information and automatically select the right way to export to
       eclipse format.

   o Export in RMS ROFF format. 
*/

  

/** 
    This function exports *one* field instance to the rms_file
    instance. It is the responsibility of the field_ROFF_export()
    function to initialize and close down the rms_file instance. 
*/

static void field_ROFF_export__(const field_type * field , rms_file_type * rms_file) {
  const int data_size             = field_config_get_volume(field->config);
  const ecl_type_enum target_type = field_config_get_ecl_type(field->config); /* Could/should in principle be input */
  const ecl_type_enum ecl_type    = field_config_get_ecl_type(field->config);
  
  void *data  = __field_alloc_3D_data(field , data_size , true ,ecl_type , target_type);
  rms_tagkey_type * data_key = rms_tagkey_alloc_complete("data" , data_size , rms_util_convert_ecl_type(target_type) , data , true);
  rms_tag_fwrite_parameter(field_config_get_ecl_kw_name(field->config) , data_key , rms_file_get_FILE(rms_file));
  rms_tagkey_free(data_key);
  free(data);
}


static rms_file_type * field_init_ROFF_export(const field_type * field, const char * filename) {
  rms_file_type * rms_file = rms_file_alloc(filename , false);
  rms_file_fopen_w(rms_file);
  rms_file_init_fwrite(rms_file , "parameter");          /* Version / byteswap ++ */
  {
    int nx,ny,nz;
    field_config_get_dims(field->config , &nx , &ny , &nz);
    rms_tag_fwrite_dimensions(nx , ny , nz , rms_file_get_FILE(rms_file));  /* Dimension header */
  }
  return rms_file;
}


static rms_file_type * field_complete_ROFF_export(const field_type * field , rms_file_type * rms_file) {
  rms_file_complete_fwrite(rms_file);
  rms_file_fclose(rms_file);
  rms_file_free(rms_file);
}




/** 
    This function exports the data of a field as a parameter to an RMS
    roff file. The export process is divided in three parts:

    1. The rms_file is opened, and initialized with some basic data
       for dimensions++
    2. The field is written to file.
    3. The file is completed / closed.

    The reason for doing it like this is that it should be easy to
    export several fields (of the same dimension+++) with repeated
    calls to 2 (i.e. field_ROFF_export__()) - that is currently not
    implemented.
*/
    
void field_ROFF_export(const field_type * field , const char * filename) {
  rms_file_type * rms_file = field_init_ROFF_export(field , filename);
  field_ROFF_export__(field , rms_file);             /* Should now be possible to several calls to field_ROFF_export__() */
  field_complete_ROFF_export(field , rms_file);
}



void field_fwrite(const field_type * field , FILE * stream) {
  const int data_size    = field_config_get_data_size(field->config);
  const int sizeof_ctype = field_config_get_sizeof_ctype(field->config);
  bool  write_compressed = field_config_write_compressed(field->config);
  
  enkf_util_fwrite_target_type(stream , FIELD);
  fwrite(&data_size               ,   sizeof  data_size        , 1 , stream);
  fwrite(&sizeof_ctype            ,   sizeof  sizeof_ctype     , 1 , stream);
  fwrite(&write_compressed        ,   sizeof  write_compressed , 1 , stream);
  if (write_compressed)
    util_fwrite_compressed(field->data , sizeof_ctype * data_size , stream);
  else
    enkf_util_fwrite(field->data    ,   sizeof_ctype , data_size , stream , __func__);
  
}


void field_ecl_write1D_fortio(const field_type * field , fortio_type * fortio , bool fmt_file , bool endian_swap ) {
  const int data_size = field_config_get_data_size(field->config);
  const ecl_type_enum ecl_type = field_config_get_ecl_type(field->config); 
  
  ecl_kw_fwrite_param_fortio(fortio , fmt_file , endian_swap , field_config_get_ecl_kw_name(field->config), ecl_type , data_size , field->data);
}


void field_ecl_write3D_fortio(const field_type * field , fortio_type * fortio , bool fmt_file , bool endian_swap ) {
  const int data_size             = field_config_get_volume(field->config);
  const ecl_type_enum target_type = field_config_get_ecl_type(field->config); /* Could/should in principle be input */
  const ecl_type_enum ecl_type    = field_config_get_ecl_type(field->config);
  void *data = __field_alloc_3D_data(field , data_size , false ,ecl_type , target_type);

  ecl_kw_fwrite_param_fortio(fortio , fmt_file , endian_swap , field_config_get_ecl_kw_name(field->config), ecl_type , data_size , data);
  free(data);
}


void field_ecl_grdecl_export(const field_type * field , FILE * stream) {
  const int data_size             = field_config_get_volume(field->config);
  const ecl_type_enum target_type = field_config_get_ecl_type(field->config); /* Could/should in principle be input */
  const ecl_type_enum ecl_type    = field_config_get_ecl_type(field->config);
  void *data                      = __field_alloc_3D_data(field , data_size , false , ecl_type , target_type );
  ecl_kw_type            * ecl_kw = ecl_kw_alloc_complete_shared(true , true , field_config_get_ecl_kw_name(field->config) , data_size , target_type , data);

  ecl_kw_fprintf_grdecl(ecl_kw , stream);
  ecl_kw_free(ecl_kw);
  free(data);

}




/**
  This is the generic "export field to eclipse" function. It will
  check up the config object to determine how to export the field,
  and then call the appropriate function. The alternatives are:

  * Restart format - only active cells (field_ecl_write1D_fortio).
  * Restart format - all cells         (field_ecl_write3D_fortio).
  * GRDECL  format                     (field_ecl_grdecl_export)
*/  

void field_export(const field_type * field, const char * file , field_file_format_type file_type) {
  if ((file_type == ecl_kw_file_all_cells) || (file_type == ecl_kw_file_active_cells)) {
    fortio_type * fortio;
    bool fmt_file , endian_swap;

    field_config_set_io_options(field->config , &fmt_file , &endian_swap);
    fortio = fortio_fopen(file , "w" , endian_swap);

    if (file_type == ecl_kw_file_all_cells)
      field_ecl_write3D_fortio(field , fortio , fmt_file , endian_swap);
    else
      field_ecl_write1D_fortio(field , fortio , fmt_file , endian_swap);

    fortio_fclose(fortio);
  } else if (file_type == ecl_grdecl_file) {
    FILE * stream = util_fopen(file , "w");
    field_ecl_grdecl_export(field , stream);
    fclose(stream);
  } else if (file_type == rms_roff_file) 
    field_ROFF_export(field , file);
  else
    util_abort("%s: internal error file_type = %d - aborting \n",__func__ , file_type);
}


void field_ecl_write(const field_type * field , const char * file) {
  field_file_format_type export_format = field_config_get_ecl_export_format(field->config);
  field_export(field , file , export_format);
}





void field_initialize(field_type *field , int iens) {
  field_init_type init_type = field_config_get_init_type(field->config);
  if (init_type & load_unique) {
    char * filename = field_config_alloc_init_file(field->config , iens);
    field_fload(field , filename , field_config_get_endian_swap(field->config));
    init_type -= load_unique;
    free(filename);
  }
  if (init_type != 0) {
    fprintf(stderr,"%s not fully implemented ... \n",__func__);
    abort();
  }
}


void field_free(field_type *field) {
  field_free_data(field);
  free(field);
}


void field_truncate(field_type * field) {
  const field_config_type *config     = field->config;
  ecl_type_enum ecl_type              = field_config_get_ecl_type(config);
  const int                data_size  = field_config_get_data_size(config);
  if (ecl_type == ecl_float_type) {
    float min_value = 0.00001;
    float max_value = 199999999.0;
    
    enkf_util_truncate(field->data , data_size , ecl_type , &min_value , &max_value);
  }
}


int field_deserialize(field_type * field , int internal_offset , size_t serial_size , const double * serial_data , size_t stride , size_t offset) {
  const field_config_type *config      = field->config;
  const int                data_size   = field_config_get_data_size(config);
  ecl_type_enum ecl_type               = field_config_get_ecl_type(config);
  int new_internal_offset;

  new_internal_offset = enkf_util_deserializeII(field->data , ecl_type , NULL , internal_offset , data_size , serial_size , serial_data , offset , stride);

  field_truncate(field);
  return new_internal_offset;
}




int field_serialize(const field_type *field , int internal_offset , size_t serial_data_size ,  double *serial_data , size_t stride , size_t offset , bool *complete) {
  const field_config_type *config     = field->config;
  ecl_type_enum ecl_type              = field_config_get_ecl_type(config);
  const int                data_size  = field_config_get_data_size(config);

  int elements_added;
  elements_added = enkf_util_serializeII(field->data , ecl_type , NULL , internal_offset , data_size , serial_data , serial_data_size , offset , stride , complete);
  return elements_added;
}




/*
  int index05D = config->index_map[ k * config->nx * config->ny + j * config->nx + i];      2
*/



void field_ijk_get(const field_type * field , int i , int j , int k , void * value) {
  int global_index = field_config_global_index(field->config , i , j , k);
  int sizeof_ctype = field_config_get_sizeof_ctype(field->config);
  memcpy(value , &field->data[global_index * sizeof_ctype] , sizeof_ctype);
}



/**
   Takes a global index (counting only active cells) as input, and
   returns a double.
*/


double field_iget_double(const field_type * field , int global_index) {
  ecl_type_enum ecl_type = field_config_get_ecl_type(field->config);
  int sizeof_ctype 	 = field_config_get_sizeof_ctype(field->config);
  char buffer[8]; /* Enough to hold one double */
  memcpy(buffer , &field->data[global_index * sizeof_ctype] , sizeof_ctype);
  if ( ecl_type == ecl_double_type ) 
    return *((double *) buffer);
  else if (ecl_type == ecl_float_type) 
    {
      double double_value;
      float  float_value;
      
      float_value  = *((float *) buffer);
      double_value = float_value;
      
      return double_value;
    }
  else {
    util_abort("%s: failed - wrong internaø type \n",__func__);
    return -1;
  }
}



void field_ijk_set(field_type * field , int i , int j , int k , const void * value) {
  int global_index = field_config_global_index(field->config , i , j , k);
  int sizeof_ctype = field_config_get_sizeof_ctype(field->config);
  memcpy(&field->data[global_index * sizeof_ctype] , value , sizeof_ctype);
}


#define INDEXED_SET_MACRO(t,s,n,index) \
{                                      \
   int i;                              \
   for (i=0; i < (n); i++)             \
       (t)[index[i]] = (s)[i];                \
}                                      \


void field_indexed_set(field_type * field, ecl_type_enum src_type , int len , const int * index_list , const void * __value_list) {
  const char * value_list = (const char *) __value_list;
  int sizeof_ctype = field_config_get_sizeof_ctype(field->config);
  ecl_type_enum target_type = field_config_get_ecl_type(field->config);

  if (src_type == target_type) {
    /* Same type */
    int i;
    for (i=0; i < len; i++) 
      memcpy(&field->data[index_list[i] * sizeof_ctype] , &value_list[i * sizeof_ctype] , sizeof_ctype);
  } else {
    switch (target_type) {
    case(ecl_float_type):
      /* double -> float */
      {
	float * field_data = (float *) field->data;
	if (src_type == ecl_double_type) {
	  double * src_data = (double *) __value_list;
	  INDEXED_SET_MACRO(field_data , src_data , len , index_list);
	} else {
	  fprintf(stderr,"%s both existing field - and indexed values must be float / double - aborting\n",__func__);
	  abort();
	}
      }
      break;
    case(ecl_double_type):
      /* float -> double  */
      {
	double * field_data = (double *) field->data;
	if (src_type == ecl_float_type) {
	  float * src_data = (float *) __value_list;
	  INDEXED_SET_MACRO(field_data , src_data , len , index_list);
	} else {
	  fprintf(stderr,"%s both existing field - and indexed values must be float / double - aborting\n",__func__);
	  abort();
	}
      }
      break;
    default:
      fprintf(stderr,"%s existing field must be of type float/double - aborting \n",__func__);
      abort();
    }
  }
}



double * field_indexed_get_alloc(const field_type * field, int len, const int * index_list)
{
  double * data = util_malloc(len * sizeof * data, __func__);
  ecl_type_enum src_type = field_config_get_ecl_type(field->config);
  int sizeof_ctype = field_config_get_sizeof_ctype(field->config);
  
  if(src_type == ecl_double_type)
  {
    /* double -> double */
    int i;
    for(i=0; i<len; i++)
      memcpy(&data[i * sizeof_ctype], &field->data[index_list[i] * sizeof_ctype] , sizeof_ctype);
  }
  else if(src_type == ecl_float_type) {
    /* float -> double */
    util_abort("%s: indexing ignored in float -> double converison \n",__func__);
    util_float_to_double(data, (float *) field->data, len);
  }
  else {
    fprintf(stderr,"%s: existing field must of type float/double - aborting. \n", __func__);
    abort();
  }

  return data;
}



bool field_ijk_valid(const field_type * field , int i , int j , int k) {
  int global_index = field_config_global_index(field->config , i , j , k);
  if (global_index >=0)
    return true;
  else
    return false;
}


void field_ijk_get_if_valid(const field_type * field , int i , int j , int k , void * value , bool * valid) {
  int global_index = field_config_global_index(field->config , i , j , k);
  if (global_index >=0) {
    *valid = true;
    field_ijk_get(field , i , j , k , value);
  } else 
    *valid = false;
}


int field_get_global_index(const field_type * field , int i , int j  , int k) {
  return field_config_global_index(field->config , i , j , k);
}



void field_copy_ecl_kw_data(field_type * field , const ecl_kw_type * ecl_kw) {
  const field_config_type * config = field->config;
  const int data_size      	   = field_config_get_data_size(config);
  ecl_type_enum field_type 	   = field_config_get_ecl_type(field->config);
  ecl_type_enum kw_type            = ecl_kw_get_type(ecl_kw);
  if (data_size != ecl_kw_get_size(ecl_kw)) {
    fprintf(stderr,"%s: fatal error - incorrect size for:%s [config:%d , file:%d] - aborting \n",__func__ , config->ecl_kw_name , data_size , ecl_kw_get_size(ecl_kw));
    abort();
  } 
  ecl_util_memcpy_typed_data(field->data , ecl_kw_get_data_ref(ecl_kw) , field_type , kw_type , ecl_kw_get_size(ecl_kw));
}



/*****************************************************************/

void field_fload_rms(field_type * field , const char * filename) {
  const char * key           = field_config_get_ecl_kw_name(field->config);
  rms_file_type * rms_file   = rms_file_alloc(filename , false);
  rms_tagkey_type * data_tag = rms_file_fread_alloc_data_tagkey(rms_file , "parameter" , "name" , key);
  ecl_type_enum   ecl_type   = rms_tagkey_get_ecl_type(data_tag);
  if (rms_tagkey_get_size(data_tag) != field_config_get_volume(field->config)) {
    fprintf(stderr,"%s: trying to import rms_data_tag from:%s with wrong size - aborting \n",__func__ , filename);
    abort();
  }
  field_import3D(field , rms_tagkey_get_data_ref(data_tag) , true , ecl_type);
  rms_tagkey_free(data_tag);
  rms_file_free(rms_file);
}



void field_fload_ecl_kw(field_type * field , const char * filename , bool endian_flip) {
  const char * key = field_config_get_ecl_kw_name(field->config);
  ecl_kw_type * ecl_kw;
  
  {
    bool fmt_file        = ecl_fstate_fmt_file(filename);
    fortio_type * fortio = fortio_fopen(filename , "r" , endian_flip);
    ecl_kw_fseek_kw(key , fmt_file , true , true , fortio);
    ecl_kw = ecl_kw_fread_alloc(fortio , false);
    fortio_fclose(fortio);
  }
  
  if (field_config_get_volume(field->config) == ecl_kw_get_size(ecl_kw)) 
    field_import3D(field , ecl_kw_get_data_ref(ecl_kw) , false , ecl_kw_get_type(ecl_kw));
  else if (field_config_get_active_size(field->config) == ecl_kw_get_size(ecl_kw)) {
    /* Keyword is already packed - e.g. from a restart file */
    ecl_type_enum field_type = field_config_get_ecl_type(field->config);
    ecl_type_enum kw_type    = ecl_kw_get_type(ecl_kw);
    ecl_util_memcpy_typed_data(field->data , ecl_kw_get_data_ref(ecl_kw) , field_type , kw_type , ecl_kw_get_size(ecl_kw));
  } else {
    fprintf(stderr,"%s: trying to import ecl_kw(%s) of wrong size: field:%d  ecl_kw:%d \n",__func__ , ecl_kw_get_header_ref(ecl_kw) , field_config_get_active_size(field->config) , ecl_kw_get_size(ecl_kw));
    abort();
  }
  ecl_kw_free(ecl_kw);
}



/* No type translation possible */
void field_fload_ecl_grdecl(field_type * field , const char * filename , bool endian_flip) {
  const char * key = field_config_get_ecl_kw_name(field->config);
  int size = field_config_get_volume(field->config);
  ecl_type_enum ecl_type = field_config_get_ecl_type(field->config);
  ecl_kw_type * ecl_kw;
  {
    FILE * stream = util_fopen(filename , "r");
    ecl_kw = ecl_kw_fscanf_alloc_grdecl_data(stream , size , ecl_type , endian_flip);
    fclose(stream);
  }

  if (strncmp(key , ecl_kw_get_header_ref(ecl_kw) , strlen(key)) != 0) {
    fprintf(stderr,"%s: did not load keyword:%s from file:%s - seek() is not implemented for grdecl files - aborting \n",__func__ , key , filename);
    abort();
  }
  
  field_import3D(field , ecl_kw_get_data_ref(ecl_kw) , false , ecl_kw_get_type(ecl_kw));
  ecl_kw_free(ecl_kw);
}



void field_fload_typed(field_type * field , const char * filename ,  bool endian_flip , field_file_format_type file_type) {
  switch (file_type) {
  case(rms_roff_file):
    field_fload_rms(field , filename );
    break;
  case(ecl_kw_file):
    field_fload_ecl_kw(field , filename  , endian_flip);
    break;
  case(ecl_grdecl_file):
    field_fload_ecl_grdecl(field , filename  , endian_flip);
    break;
  default:
    fprintf(stderr,"%s: file_type:%d not recognized - aborting \n",__func__ , file_type);
    abort();
  }
}




void field_fload(field_type * field , const char * filename , bool endian_flip) {
  field_file_format_type file_type = field_config_guess_file_type(filename , endian_flip);
  if (file_type == unknown_file) file_type = field_config_manual_file_type(filename , true);
  field_fload_typed(field , filename , endian_flip , file_type);
}



void field_fload_auto(field_type * field , const char * filename , bool endian_flip) {
  field_file_format_type file_type = field_config_guess_file_type(filename , endian_flip);
  field_fload_typed(field , filename , endian_flip , file_type);
}



/**
   This function compares two fields, and return true if they are
   equal. Observe that the config comparison is done with plain
   pointer comparison, i.e. the actual content of the config objects
   is not compared. If the two fields point to different config
   objects, the comparision will fail immediately - without checking the
   content of the fields.
*/

bool field_cmp(const field_type * f1 , const field_type * f2) {
  if (f1->config != f2->config) {
    fprintf(stderr,"The two fields have different config objects - and the comparison fails trivially.\n");
    return false;
  } else {
    const int byte_size = field_config_get_byte_size(f1->config);   
    if (memcmp( f1->data , f2->data , byte_size) != 0)
      return false;
    else
      return true;
  }
}


/*****************************************************************/


/* /\* Skal param_name vaere en variabel ?? *\/ */
/* void field_rms_export_parameter(const field_type * field , const char * param_name , const float * data3D,  const rms_file_type * rms_file) { */
/*   const field_config_type * config = field->config; */
/*   const int data_size = field_config_get_data_size(config); */
  
/*   /\* Hardcoded rms_float_type *\/ */
/*   rms_tagkey_type *tagkey = rms_tagkey_alloc_complete("data" , data_size , rms_float_type , data3D , true); */
/*   rms_tag_fwrite_parameter(param_name , tagkey , rms_file_get_FILE(rms_file)); */
/*   rms_tagkey_free(tagkey); */
  
/* } */




void field_get_dims(const field_type * field, int *nx, int *ny , int *nz) {
  field_config_get_dims(field->config , nx , ny ,nz);
}


void field_apply_limits(field_type * field) {
  field_config_apply_limits(field->config , field->data);
}


/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/

MATH_OPS(field)
VOID_ALLOC(field)
VOID_FREE(field)
VOID_FREE_DATA(field)
VOID_REALLOC_DATA(field)
VOID_ECL_WRITE (field)
VOID_FWRITE (field)
VOID_FREAD  (field)
VOID_COPYC     (field)
VOID_SERIALIZE (field);
VOID_DESERIALIZE (field);
VOID_INITIALIZE(field);
VOID_CLEAR(field);
ENSEMBLE_MULX_VECTOR(field);




