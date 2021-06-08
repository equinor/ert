/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'rms_util.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <ert/util/util.hpp>

#include <ert/rms/rms_util.hpp>


/*
  This translates from the RMS data layout to "Fortan / ECLIPSE" data
  layout.

  RMS: k index is running fastest *AND* backwards.
  F90: i is running fastest, and k is running the 'normal' way.

  This function should be *THE ONLY* place in the code where explicit mention
  is made to the RMS ordering sequence.
*/


int rms_util_global_index_from_eclipse_ijk(int nx, int ny , int nz , int i , int j , int k) {
  return i*ny*nz  +  j*nz  +  (nz - k - 1);
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
  long int init_pos = util_ftell(stream);
  int len;
  rms_util_fskip_string(stream);
  len = util_ftell(stream) - init_pos;
  fseek(stream , init_pos , SEEK_SET);
  return len;
}


/*
  max_length *includes* the trailing \0.
*/
bool rms_util_fread_string(char *string , int max_length , FILE *stream) {
  bool read_ok = true;
  bool cont    = true;
  long int init_pos = util_ftell(stream);
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


rms_type_enum rms_util_convert_ecl_type(ecl_data_type data_type) {
  rms_type_enum rms_type = rms_int_type;  /* Shut up the compiler */
  switch (ecl_type_get_type(data_type)) {
  case(ECL_INT_TYPE):
    rms_type = rms_int_type;
    break;
  case(ECL_FLOAT_TYPE):
    rms_type = rms_float_type;
    break;
  case(ECL_DOUBLE_TYPE):
    rms_type = rms_double_type;
    break;
  default:
    util_abort("%s: Conversion ecl_type -> rms_type not supported for ecl_type:%s \n",__func__ , ecl_type_alloc_name(data_type));
  }
  return rms_type;
}
