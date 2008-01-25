#include <stdlib.h>
#include <string.h>
#include <enkf_types.h>
#include <util.h>
#include <field_config.h>
#include <enkf_macros.h>
#include <ecl_kw.h>
#include <ecl_util.h>
#include <rms_file.h>


/*****************************************************************/


void field_config_set_ecl_kw_name(field_config_type * config , const char * ecl_kw_name) {
  config->ecl_kw_name = util_realloc_string_copy(config->ecl_kw_name , ecl_kw_name);
}

void field_config_set_ecl_type(field_config_type * config , ecl_type_enum ecl_type) {
  config->ecl_type     = ecl_type;
  config->sizeof_ctype = ecl_util_get_sizeof_ctype(ecl_type);
}


static const char * field_config_file_type_string(field_file_type file_type) {
  switch (file_type) {
  case(rms_roff_file):
    return "Binary ROFF file from RMS";
    break;
  case(ecl_kw_file):
    return "ECLIPSE file in restart format";
    break;
  case(ecl_grdecl_file):
    return "ECLIPSE file in grdecl format";
    break;
  default:
    fprintf(stderr,"%s: invalid file type \n",__func__);
    abort();
  }
}


static bool field_config_valid_file_type(field_file_type file_type) {
  if (file_type == rms_roff_file || file_type == ecl_kw_file || file_type == ecl_grdecl_file)
    return true;
  else
    return false;
}




field_file_type field_config_manual_file_type(const char * filename) {
  field_file_type file_type = unknown_file;
  printf("\nCould not determine type of file: %s \n",filename);
  do {
    printf("----------------------------------------------------------------\n");
    printf(" %3d: %s \n",rms_roff_file , field_config_file_type_string(rms_roff_file));
    printf(" %3d: %s \n",ecl_kw_file , field_config_file_type_string(ecl_kw_file));
    printf(" %3d: %s \n",ecl_grdecl_file , field_config_file_type_string(ecl_grdecl_file));
    printf("----------------------------------------------------------------\n\n");
    {
      int int_file_type;
      printf("===> "); scanf("%d" , &int_file_type); file_type = int_file_type;
    }
    if (!field_config_valid_file_type(file_type))
      file_type = unknown_file;
  } while(file_type == unknown_file);
  return file_type;
}



field_file_type field_config_guess_file_type(const char * filename , bool endian_flip) {
  bool fmt_file = util_fmt_bit8(filename );
  FILE * stream = util_fopen(filename , "r");

  field_file_type file_type;
  if (ecl_kw_is_kw_file(stream , fmt_file , endian_flip))
    file_type = ecl_kw_file;
  else if (rms_file_is_roff(stream))
    file_type = rms_roff_file;
  else if (ecl_kw_is_grdecl_file(stream))  /* This is the weakest test - and should be last in a cascading if / else hierarchy. */
    file_type = ecl_grdecl_file;
  else 
    file_type = unknown_file;              /* MUST Check on this return value */

  fclose(stream);
  return file_type;
}



field_config_type * field_config_alloc(const char * ecl_kw_name , ecl_type_enum ecl_type , int nx , int ny , int nz , int active_size , const int * index_map , int logmode) {
  field_config_type *config = malloc(sizeof *config);
  
  /*
    Observe that size is the number of *ACTIVCE* cells,
    and generally *not* equal to nx*ny*nz.
  */
  config->data_size        = active_size; 

  
  config->ecl_kw_name = NULL;
  field_config_set_ecl_kw_name(config , ecl_kw_name);
  field_config_set_ecl_type(config , ecl_type);
  config->logmode = logmode;

  config->nx = nx;
  config->ny = ny;
  config->nz = nz;

  config->sx = 1;
  config->sy = nx;
  config->sz = nx * ny;
  config->index_map = index_map;
  
  config->fmt_file    	   = false;
  config->endian_swap 	   = true;
  config->limits_set  	   = false;
  config->min_value   	   = malloc(config->sizeof_ctype);
  config->max_value   	   = malloc(config->sizeof_ctype);
  config->write_compressed = true;
  return config;
}


bool field_config_write_compressed(const field_config_type * config) { return config->write_compressed; }


void field_config_set_limits(field_config_type * config , void * min_value , void * max_value) {
  memcpy(config->min_value , min_value , config->sizeof_ctype);
  memcpy(config->max_value , max_value , config->sizeof_ctype);
  config->limits_set = true;
}


void field_config_apply_limits(const field_config_type * config, void * _data) {
  if (config->ecl_type != ecl_double_type) {
    fprintf(stderr,"%s: sorry - limits only implemented for double fields currently \n",__func__);
    abort();
  }

  if (config->limits_set) {
    switch (config->ecl_type) {
    case(ecl_double_type):
      {
	double *data = (double *) _data;
	int i;
	for (i = 0; i < config->data_size; i++) 
	  util_apply_double_limits(&data[i] , *((double *) config->min_value) , *((double *) config->max_value));
      }
      break;
    case(ecl_float_type):
      {
	float *data = (float *) _data;
	int i;
	for (i = 0; i < config->data_size; i++) 
	  util_apply_float_limits(&data[i] , *((float *) config->min_value) , *((float *) config->max_value));
      }
      break;
    case(ecl_int_type):
      {
	int *data = (int *) _data;
	int i;
	for (i = 0; i < config->data_size; i++) 
	  util_apply_int_limits(&data[i] , *((int *) config->min_value) , *((int *) config->max_value));
      }
      break;
    default:
      fprintf(stderr,"%s field limits only applied for int/double/float fields - aborting \n",__func__);
      abort();
    }
  } else {
    fprintf(stderr,"%s: must set limits with a call to : field_config_set_limits() first - aborting \n",__func__);
    abort();
  }
}




void field_config_set_io_options(const field_config_type * config , bool *fmt_file , bool * endian_swap) {
  *fmt_file    = config->fmt_file;
  *endian_swap = config->endian_swap;
  /*
   *ecl_type    = config->ecl_type;
   */
}


void field_config_free(field_config_type * config) {
  free(config->min_value);
  free(config->max_value);
  if (config->ecl_kw_name != NULL) free(config->ecl_kw_name);
  free(config);
}
  

int field_config_get_volume(const field_config_type * config) {
  return config->nx * config->ny * config->nz;
}

ecl_type_enum field_config_get_ecl_type(const field_config_type * config) {
  return config->ecl_type;
}

int field_config_get_byte_size(const field_config_type * config) {
  return config->data_size * config->sizeof_ctype;
}

int field_config_get_active_size(const field_config_type * config) {
  return config->data_size;
}

int field_config_get_sizeof_ctype(const field_config_type * config) { return config->sizeof_ctype; }



/*
  Observe that the indices are zero-based, in contrast 
  to those used by eclipse which are based on one.
*/

inline int field_config_global_index(const field_config_type * config , int i , int j , int k) {
  return config->index_map[ k * config->nx * config->ny + j * config->nx + i];
}


void field_config_get_ijk(const field_config_type * config , int global_index, int *_i , int *_j , int *_k) {
  int i,j,k;
  if (global_index >= config->data_size || global_index < 0) {
    fprintf(stderr,"%s: global_index: %d is not in intervale [0,%d) - aborting \n",__func__ , global_index , config->data_size);
    abort();
  }

  for (k=0; k < config->nz; k++)
    for (j=0; j < config->ny; j++)
      for (i=0; i < config->nx; i++)
	if (field_config_global_index(config , i,j,k) == global_index) {
	  *_i = i;
	  *_j = j;
	  *_k = k;
	  return;
	}
}




void field_config_get_dims(const field_config_type * config , int *nx , int *ny , int *nz) {
  *nx = config->nx;
  *ny = config->ny;
  *nz = config->nz;
}


/*****************************************************************/
CONFIG_GET_ECL_KW_NAME(field);
GET_DATA_SIZE(field)
VOID_FREE(field_config)
