#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <rms_util.h>


/*
  This translates from the RMS data layout to "fortan" data layout.

  RMS: k index is running fastest *AND* backwards.
  F90: i is running fastest, and k is running the 'normal' way.
  
*/

void rms_util_set_fortran_data(void *_f90_data , const void * _rms_data, int sizeof_ctype , int nx, int ny , int nz) {
  char *f90_data       = (char *)       _f90_data;
  const char *rms_data = (const char *) _rms_data;
  int i,j,k,rms_index, f90_index;
  rms_index = -1;
  for (i=0; i < nx; i++) 
    for (j=0; j < ny; j++)
      for (k= (nz -1); k >= 0; k--) {
	rms_index += 1;
	f90_index  = i + j*nx + k*nx*ny;
	memcpy(&f90_data[f90_index * sizeof_ctype] , &rms_data[rms_index * sizeof_ctype] , sizeof_ctype);
      }
}



void rms_util_read_fortran_data(const void *_f90_data , void * _rms_data, int sizeof_ctype , int nx, int ny , int nz) {
  const char *f90_data = (const char *) _f90_data;
  char *rms_data       = (char *)       _rms_data;
  int i,j,k,rms_index, f90_index;
  rms_index = -1;
  for (i=0; i < nx; i++) 
    for (j=0; j < ny; j++)
      for (k= (nz -1); k >= 0; k--) {
	rms_index += 1;
	f90_index = i + j*nx + k*nx*ny;
	memcpy(&rms_data[rms_index * sizeof_ctype] , &f90_data[f90_index * sizeof_ctype] , sizeof_ctype);
      }
}



void rms_util_translate_undef(void * _data , int size , int sizeof_ctype , const void * old_undef , const void * new_undef) {
  char * data = (char *) _data;
  int i;
  for (i=0; i < size; i++) {
    if (memcmp( &data[i*sizeof_ctype] , old_undef , sizeof_ctype) == 0)
      memcpy( &data[i*sizeof_ctype] , new_undef , sizeof_ctype);
  }
}


void rms_util_fskip_string(FILE *stream) {
  char c;
  bool cont = true;
  while (cont) {
    fread(&c , 1 , 1 , stream);
    if (c == 0)
      cont = false;
  } 
}


int rms_util_fread_strlen(FILE *stream) {
  long int init_pos = ftell(stream);
  int len;
  rms_util_fskip_string(stream);
  len = ftell(stream) - init_pos;
  fseek(stream , init_pos , SEEK_SET);
  return len;
}

/*
  max_length *includes* the trailing \0.
*/
bool rms_util_fread_string(char *string , int max_length , FILE *stream) {
  bool read_ok = true;
  bool cont    = true;
  long int init_pos = ftell(stream);
  int pos = 0;
  while (cont) {
    fread(&string[pos] , sizeof *string , 1 , stream);
    if (string[pos] == 0) {
      read_ok = true;
      cont = false;
    } else {
      pos++;
      if (max_length > 0) {
	if (pos == max_length) {
	  read_ok = false;
	  fseek(stream , init_pos , SEEK_SET);
	  cont = false;
	}
      }
    } 
  } 
  
  return read_ok;
}


void rms_util_fwrite_string(const char * string , FILE *stream) {
  fwrite(string , sizeof * string , strlen(string) , stream);
  fputc('\0' , stream);
}

void rms_util_fwrite_comment(const char * comment , FILE *stream) {
  fputc('#' , stream);
  fwrite(comment , sizeof * comment , strlen(comment) , stream);
  fputc('#' , stream);
  fputc('\0' , stream);
}


void rms_util_fwrite_newline(FILE *stream) {
  return;
}


