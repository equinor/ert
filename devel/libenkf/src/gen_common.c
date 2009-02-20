#include <stdlib.h>
#include <util.h>
#include <stdio.h>
#include <string.h>
#include <gen_data_config.h>
#include <gen_common.h>
#include <fortio.h>
#include <ecl_util.h>

/**
   This file implements some (very basic) functionality which is used
   by both the gen_data and gen_obs objects.
*/


void * gen_common_fscanf_alloc(const char * file , ecl_type_enum load_type , int * size) {
  FILE * stream    	  = util_fopen(file , "r");
  int sizeof_ctype        = ecl_util_get_sizeof_ctype(load_type);
  int buffer_elements     = *size;
  int current_size        = 0;
  int fscanf_return       = 1; /* To keep the compiler happy .*/
  char * buffer ;
  
  if (buffer_elements == 0)
    buffer_elements = 100;
  
  buffer = util_malloc( buffer_elements * sizeof_ctype , __func__);
  {
    do {
      if (load_type == ecl_float_type) {
	float  * float_buffer = (float *) buffer;
	fscanf_return = fscanf(stream , "%g" , &float_buffer[current_size]);
      } else if (load_type == ecl_double_type) {
	double  * double_buffer = (double *) buffer;
	fscanf_return = fscanf(stream , "%lg" , &double_buffer[current_size]);
      } else if (load_type == ecl_int_type) {
	int * int_buffer = (int *) buffer;
	fscanf_return = fscanf(stream , "%d" , &int_buffer[current_size]);
      }  else 
	util_abort("%s: god dammit - internal error \n",__func__);
      
      if (fscanf_return == 1)
	current_size += 1;
      
      if (current_size == buffer_elements) {
	buffer_elements *= 2;
	buffer = util_realloc( buffer , buffer_elements * sizeof_ctype , __func__);
      }
    } while (fscanf_return == 1);
  }
  if (fscanf_return != EOF) 
    util_abort("%s: scanning of %s terminated before EOF was reached -- fix your file.\n" , __func__ , file);
  
  fclose(stream);
  *size = current_size;
  return buffer;
}



void * gen_common_fread_alloc(const char * file , ecl_type_enum load_type , int * size) {
  const int max_read_size = 100000;
  FILE * stream    	  = util_fopen(file , "r");
  int sizeof_ctype        = ecl_util_get_sizeof_ctype(load_type);
  int read_size           = 4096; /* Shot in the wild */
  int current_size        = 0;
  int buffer_elements;
  int fread_return;
  char * buffer;
  
  
  buffer_elements = read_size;
  buffer = util_malloc( buffer_elements * sizeof_ctype , __func__);
  {
    do {
      fread_return  = fread( &buffer[ current_size * sizeof_ctype] , sizeof_ctype , read_size , stream);
      current_size += fread_return;
      
      if (!feof(stream)) {
	/* Allocate more elements. */
	if (current_size == buffer_elements) {
	  read_size *= 2;
	  read_size = util_int_min(read_size , max_read_size);
	  buffer_elements += read_size;
	  buffer = util_realloc( buffer , buffer_elements * sizeof_ctype , __func__);
	} else 
	  util_abort("%s: internal error ?? \n",__func__);
      }
    } while (!feof(stream));
  }
  *size = current_size;
  return buffer;
}


/*
  If the load_format is binary_float or binary_double, the ASCII_type
  is *NOT* consulted. The load_type is set to float/double depending
  on what was actually used when the data was loaded.
*/

void * gen_common_fload_alloc(const char * file , gen_data_file_format_type load_format , ecl_type_enum ASCII_type , ecl_type_enum * load_type , int * size) { 
  void * buffer = NULL;

  if (load_format == ASCII) {
    *load_type = ASCII_type;
    buffer =  gen_common_fscanf_alloc(file , ASCII_type , size);
  } else if (load_format == binary_float) {
    *load_type = ecl_float_type;
    buffer = gen_common_fread_alloc(file , ecl_float_type , size);
  } else if (load_format == binary_double) {
    *load_type = ecl_double_type;
    buffer = gen_common_fread_alloc(file , ecl_double_type , size);
  } else 
    util_abort("%s: trying to load with unsupported format ... \n");
  
  //{
  //  FILE * stream = util_fopen("/tmp/data.txt" , "w");
  //  double * data = (double *) buffer;
  //  for (int i = 0; i < *size; i++)
  //   fprintf(stream , "%12.4f[%06d] \n",data[i] , i);
  //  fclose(stream);
  //}
  
  return buffer;
}
