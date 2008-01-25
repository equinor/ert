#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <enkf_config.h>
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
#include <fortio.h>

#define  DEBUG
#define  TARGET_TYPE FIELD
#include "enkf_debug.h"



GET_DATA_SIZE_HEADER(field);

/*****************************************************************/

struct field_struct {
  DEBUG_DECLARE
  const  field_config_type * config;
  char  *data;
  
  bool   shared_data;
  int    shared_byte_size;
};



#define EXPORT_MACRO                                                                           \
for (k=0; k < config->nz; k++) {                                                               \
  for (j=0; j < config->ny; j++) {                                                             \
    for (i=0; i < config->nx; i++) {                                                           \
      int index1D = field_config_global_index(config , i , j , k);                             \
      int index3D;                                                                             \
      if (rms_order)                                               		     	       \
        index3D = i * config->ny*config->nz  +  j * config->ny + (config->nz - k);             \
      else                                                                       	       \
        index3D = i + j * config->nx + k* config->nx*config->ny;           	               \
      if (index1D >= 0)                                                                        \
	target_data[index3D] = src_data[index1D];                               	       \
      else                                                                                     \
        memcpy(&target_data[index3D] , fill_value , sizeof_ctype_target);                      \
     }                                                                                         \
  }                                                                                            \
}


void field_export3D(const field_type * field , void *_target_data , bool rms_order , ecl_type_enum target_type , void *fill_value) {
  const field_config_type * config = field->config;
  int   sizeof_ctype_target = ecl_util_get_sizeof_ctype(target_type);
  int i,j,k;

  
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
  

/*****************************************************************/


#define IMPORT_MACRO                                                                           \
for (k=0; k < config->nz; k++) {                                                               \
  for (j=0; j < config->ny; j++) {                                                             \
    for (i=0; i < config->nx; i++) {                                                           \
      int index1D = field_config_global_index(config , i , j , k);                                        \
      int index3D;                                                                             \
      if (index1D >= 0) {                                                                      \
	if (rms_order)                                               		     	       \
	  index3D = i * config->ny*config->nz  +  j * config->ny + (config->nz - k);           \
	else                                                                       	       \
	  index3D = i + j * config->nx + k* config->nx*config->ny;           	               \
	target_data[index1D] = src_data[index3D] ;                               	       \
     }                                                                                         \
   }                                                                                           \
  }                                                                                            \
}



static void field_import3D(field_type * field , const void *_src_data , bool rms_order , ecl_type_enum src_type) {
  const field_config_type * config = field->config;
  int i,j,k;

  
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
  enkf_util_fread_assert_target_type(stream , FIELD , __func__);
  fread(&data_size     	  , sizeof  data_size        , 1 , stream);
  fread(&sizeof_ctype 	  , sizeof  sizeof_ctype     , 1 , stream);
  fread(&read_compressed  , sizeof  read_compressed  , 1 , stream);
  if (read_compressed)
    util_fread_compressed(field->data , stream);
  else
    enkf_util_fread(field->data , sizeof_ctype , data_size , stream , __func__);
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

  void *data;
  data = enkf_util_malloc(data_size * ecl_util_get_sizeof_ctype(target_type) , __func__);
  if (ecl_type == ecl_double_type) {
    double fill = 0.0;
    field_export3D(field , data , false , target_type , &fill);
  } else if (ecl_type == ecl_float_type) {
    float fill = 0.0;
    field_export3D(field , data , false , target_type , &fill);
  } else if (ecl_type == ecl_int_type) {
    int fill = 0;
    field_export3D(field , data , false , target_type , &fill);
  } else {
    fprintf(stderr,"%s: trying to export type != int/float/double - aborting \n",__func__);
    abort();
  }
  
  ecl_kw_fwrite_param_fortio(fortio , fmt_file , endian_swap , field_config_get_ecl_kw_name(field->config), ecl_type , data_size , data);
  free(data);
}


void field_ecl_write_allD(const field_type * field  , const char * eclfile , bool write3D) {
  fortio_type * fortio;
  bool fmt_file , endian_swap;

  field_config_set_io_options(field->config , &fmt_file , &endian_swap);
  fortio = fortio_open(eclfile , "w" , endian_swap);

  if (write3D)
    field_ecl_write3D_fortio(field , fortio , fmt_file , endian_swap );
  else
    field_ecl_write1D_fortio(field , fortio , fmt_file , endian_swap );

  fortio_close(fortio);
}




void field_ecl_write3D(const field_type * field , const char * path) {
  field_ecl_write_allD(field , path , true);
}


void field_ecl_write1D(const field_type * field , const char * path) {
  field_ecl_write_allD(field , path , false);
}



void field_ecl_write(const field_type * field , const char * path) {
  field_ecl_write1D(field , path);
}



void field_swapout(field_type * field , FILE * stream) {
  field_fwrite(field , stream);
  field_free_data(field);
}



void field_swapin(field_type * field , FILE * stream) {
  field_realloc_data(field);
  field_fread(field  , stream);
}






void field_sample(field_type *field) {
  printf("%s: Warning not implemented ... \n",__func__);
}




void field_free(field_type *field) {
  field_free_data(field);
  free(field);
}



int field_deserialize(const field_type * field , int internal_offset , size_t serial_size , const double * serial_data , size_t stride , size_t offset) {
  const field_config_type *config      = field->config;
  const int                data_size   = field_config_get_data_size(config);
  ecl_type_enum ecl_type               = field_config_get_ecl_type(config);
  double *data;
  int new_internal_offset;

  if (ecl_type == ecl_double_type)
    data = &((double *) field->data)[internal_offset];
  else if (ecl_type == ecl_float_type) 
    data = enkf_util_malloc(serial_size * sizeof * data , __func__);
  else {
    fprintf(stderr,"%s: tried to deserialize field with type:%d different from float/double - aborting \n",__func__ , ecl_type);
    abort();
  }
  new_internal_offset = enkf_util_deserialize(data , NULL , internal_offset , data_size , serial_size , serial_data , offset , stride);
  if (ecl_type == ecl_float_type) {
    util_double_to_float( &((float *) field->data)[internal_offset] , data , serial_size);
    free(data);
  }

  return new_internal_offset;
}




int field_serialize(const field_type *field , int internal_offset , size_t serial_data_size ,  double *serial_data , size_t stride , size_t offset , bool *complete) {
  const field_config_type *config     = field->config;
  ecl_type_enum ecl_type              = field_config_get_ecl_type(config);
  const int                data_size  = field_config_get_data_size(config);

  int elements_added;
  double *data;
  
  if (ecl_type == ecl_double_type)
    data = (double *) field->data;
  else if (ecl_type == ecl_float_type) {
    data = enkf_util_malloc(data_size * sizeof * data , __func__);
    util_float_to_double(data , (const float *) field->data , data_size);
  } else {
    fprintf(stderr,"%s: tried to serialize field with type:%d different from float/double - aborting \n",__func__ , ecl_type);
    abort();
  }
  elements_added = enkf_util_serialize(data , NULL , internal_offset , data_size , serial_data , serial_data_size , offset , stride , complete);
  
  if (ecl_type == ecl_float_type) free(data);
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



void field_ijk_set(field_type * field , int i , int j , int k , const void * value) {
  int global_index = field_config_global_index(field->config , i , j , k);
  int sizeof_ctype = field_config_get_sizeof_ctype(field->config);
  memcpy(&field->data[global_index * sizeof_ctype] , value , sizeof_ctype);
}


#define INDEXED_SET_MACRO(t,s,n,index) \
{                                      \
   int i;                              \
   for (i=0; i < (n); i++)             \
       (t)[i] = (s)[i];                \
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
  const char * key = field_config_get_ecl_kw_name(field->config);
  rms_file_type * rms_file   = rms_file_alloc(filename , false);
  rms_tagkey_type * data_tag = rms_file_fread_alloc_data_tagkey(rms_file , "parameter" , "name" , key);
  ecl_type_enum   ecl_type;
  {
    rms_type_enum   rms_type   = rms_tagkey_get_rms_type(data_tag);
    switch (rms_type) {
    case(rms_float_type):
      ecl_type = ecl_float_type;
      break;
    case(rms_double_type):
      ecl_type = ecl_double_type;
      break;
    case(rms_int_type):
      ecl_type = ecl_int_type;
      break;
    default:
      fprintf(stderr,"%s: sorry rms_type: %d not implemented - aborting \n",__func__ , rms_type);
      abort();
    }
  }
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
    fortio_type * fortio = fortio_open(filename , "r" , endian_flip);
    ecl_kw_fseek_kw(key , fmt_file , true , true , fortio);
    ecl_kw = ecl_kw_fread_alloc(fortio , false);
    fortio_close(fortio);
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



void field_fload_typed(field_type * field , const char * filename ,  bool endian_flip , field_file_type file_type) {
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
  field_file_type file_type = field_config_guess_file_type(filename , endian_flip);
  if (file_type == unknown_file) file_type = field_config_manual_file_type(filename);
  field_fload_typed(field , filename , endian_flip , file_type);
}


/*****************************************************************/


/* Skal param_name vaere en variabel ?? */
void field_rms_export_parameter(const field_type * field , const char * param_name , const float * data3D,  const rms_file_type * rms_file) {
  const field_config_type * config = field->config;
  const int data_size = field_config_get_data_size(config);
  
  /* Hardcoded rms_float_type */
  rms_tagkey_type *tagkey = rms_tagkey_alloc_complete("data" , data_size , rms_float_type , data3D , true);
  rms_tag_fwrite_parameter(param_name , tagkey , rms_file_get_FILE(rms_file));
  rms_tagkey_free(tagkey);
  
}

void field_get_dims(const field_type * field, int *nx, int *ny , int *nz) {
  field_config_get_dims(field->config , nx , ny ,nz);
}


void field_apply_limits(field_type * field) {
  field_config_apply_limits(field->config , field->data);
}


MATH_OPS(field)
VOID_ALLOC(field)
VOID_FREE(field)
VOID_FREE_DATA(field)
VOID_REALLOC_DATA(field)
VOID_ECL_WRITE (field)
VOID_FWRITE (field)
VOID_FREAD  (field)
VOID_COPYC     (field)
VOID_SWAPIN(field)
VOID_SWAPOUT(field)
VOID_SERIALIZE (field);
VOID_DESERIALIZE (field);
ENSEMBLE_MULX_VECTOR(field);
/******************************************************************/
/* Anonumously generated functions used by the enkf_node object   */
/******************************************************************/

VOID_FUNC      (field_clear        , field_type)
VOID_FUNC      (field_sample       , field_type)



