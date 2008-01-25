#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <rms_file.h>
#include <rms_util.h>

static rms_file_type *STATIC_RMS_FILE = NULL;

/*****************************************************************/

void old_rms_roff_load(const char *filename , const char *param_name , float *param);





void old_rms_inter_roff_param__(const char *__filename , const int *strlen, const char *__param_name , const int * param_len, float *param) {
  char *filename   = util_alloc_cstring(__filename   , strlen);
  char *param_name = util_alloc_cstring(__param_name , param_len);
  old_rms_roff_load(filename , param_name , param);
  free(filename);
  free(param_name);
}



void rms_inter_roff_param__(const char *__filename , const int *strlen, const char *__param_name , const int * param_len, 
			    const int *nx , const int *ny , const int *nz , 
			    float *undef_rms , float *undef_out,
			    float *param, float *work) {
  char *filename   = util_alloc_cstring(__filename   , strlen);
  char *param_name = util_alloc_cstring(__param_name , param_len);

  old_rms_roff_load(filename , param_name , work);
  {
    int i;
    for (i = 0; i < (*nx) * (*ny) * (*nz); i++)
      if (work[i] == *undef_rms)
	work[i] = *undef_out;
  }
  rms_util_set_fortran_data(param , work , sizeof *work , *nx , *ny , *nz);

  free(filename);
  free(param_name);
}



void rms_inter_roff_param2__(const char *__filename , const int *strlen, const char *__param_name , const int * param_len, 
			     const int *nx , const int *ny , const int *nz , 
			     float *undef_rms , float *undef_out,
			     float *param, float *work) {
  char *filename   = util_alloc_cstring(__filename   , strlen);
  char *param_name = util_alloc_cstring(__param_name , param_len);

  rms_file_type *rms_file = rms_file_alloc(filename , false);
  rms_file_fread(rms_file);
  rms_file_assert_dimensions(rms_file , *nx , *ny , *nz);
  
  {
    rms_tag_type *tag       = rms_file_get_tag_ref(rms_file , "parameter" , "name" , param_name , true);
    rms_tagkey_type *tagkey = rms_tag_get_key(tag , "data");
    work                    = rms_tagkey_get_data_ref(tagkey);
  }
  
  {
    int i;
    for (i = 0; i < (*nx) * (*ny) * (*nz); i++)
      if (work[i] == *undef_rms)
	work[i] = *undef_out;
  }
  rms_util_set_fortran_data(param , work , sizeof *work , *nx , *ny , *nz);
  rms_file_free(rms_file);
  
  free(filename);
  free(param_name);
}


void rms_init__(const char *__filename , const int *strlen,
		const int *nx , const int *ny , const int *nz) {
  char *filename   = util_alloc_cstring(__filename   , strlen);
  FILE *stream;
  STATIC_RMS_FILE  = rms_file_alloc(filename , false);
  stream = rms_file_fopen_w(STATIC_RMS_FILE);
  rms_file_init_fwrite(STATIC_RMS_FILE , "parameter");
  rms_file_add_dimensions(STATIC_RMS_FILE , *nx , *ny , *nz , true);
  free(filename);
}


static void rms_close(rms_file_type * rms_file) {
  rms_file_complete_fwrite(rms_file);
  rms_file_fclose(rms_file);
  rms_file_free(rms_file);
  rms_file = NULL;
}


void rms_close__() {
  rms_close(STATIC_RMS_FILE);
}


static void rms_add_param(rms_file_type * rms_file , const char *param_name ,
			  float undef_rms , float undef_out,
			  const float *param, float *work) {
  int dims[3];
  int size;
  rms_file_get_dims(rms_file , dims);
  rms_util_read_fortran_data(param , work , sizeof *work , dims[0] , dims[1] , dims[2]);
  size = dims[0]*dims[1]*dims[2];
  {
    int i;
    for (i = 0; i < size; i++)
      if (work[i] == undef_rms)
	work[i] = undef_out;
  }
  {
    rms_tagkey_type *tagkey = rms_tagkey_alloc_complete("data" , size , rms_float_type , work , true);
    rms_tag_fwrite_parameter(param_name , tagkey , rms_file_get_FILE(rms_file));
    rms_tagkey_free(tagkey);
  }
}


void rms_add_param__(const char *__param_name , const int * param_len,
		     float *undef_rms , float * undef_out,
		     const float *param, float *work) {

  char *param_name = util_alloc_cstring(__param_name , param_len);
  rms_add_param(STATIC_RMS_FILE , param_name , *undef_rms , *undef_out , param , work);
  free(param_name);
}



void rms_save_param__(const char *__filename , const int *strlen, const char *__param_name , const int * param_len, 
		      const int *nx , const int *ny , const int *nz , 
		      float *undef_rms , float *undef_out,
		      const float *param, float *work) {
  char *filename   = util_alloc_cstring(__filename   , strlen);
  char *param_name = util_alloc_cstring(__param_name , param_len);
  FILE *stream;

  rms_file_type *rms_file = rms_file_alloc(filename , false);

  rms_util_read_fortran_data(param , work , sizeof *work , *nx , *ny , *nz);

  
  stream = rms_file_fopen_w(rms_file);

  rms_file_init_fwrite(rms_file , "parameter" );
  rms_tag_fwrite_dimensions(*nx , *ny , *nz , stream);
  rms_add_param(rms_file , param_name , *undef_rms , *undef_out , param , work);
  rms_close(rms_file);
  
  free(filename);
  free(param_name);
}


void rms_file_2eclipse__(const char * _rms_file , const int * rms_file_len , 
			 const char * _ecl_path , const int * ecl_path_len , 
			 const int * ecl_fmt_file_int,
			 const int * ecl_endian_convert_int,
			 const int * ecl_file_nr) {
  char * rms_file   	  = util_alloc_cstring(_rms_file , rms_file_len);
  char * ecl_path   	  = util_alloc_cstring(_ecl_path , ecl_path_len);
  bool ecl_fmt_file 	  = util_intptr_2bool(ecl_fmt_file_int);
  bool ecl_endian_convert = util_intptr_2bool(ecl_endian_convert_int);
  

  rms_file_2eclipse(rms_file , ecl_path , ecl_fmt_file , ecl_endian_convert , *ecl_file_nr);

  free(rms_file);
  free(ecl_path);
}
