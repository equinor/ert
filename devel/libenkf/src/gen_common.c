#include <stdlib.h>
#include <util.h>
#include <stdio.h>
#include <string.h>
#include <gen_common.h>
#include <fortio.h>
#include <ecl_util.h>

/**
   This file implements some (very basic) functionality which is
   common to both the gen_data and gen_obs objects.
*/



/* 
  This functions reads the start of the file, and determines which
  type of file it is. The different filetypes are recognized as
  follows:

  1. fortran_binary - this is recognized by a fortio function, which
     looks for the special fortran record-start/record-end
     markers. Altough these markers uniquely identify a binary file,
     we stille require the file to start with "BINARY".

  2. ASCII files start with the string "ASCII".

  3. Binary files start with the string "BINARY\0".

  The magic strings ASCII / BINARY are NOT case sensitive. Observe
  that the BINARY string MUST contain the terminating \0; this is
  mainly to discipline the user when it comes to the TAG, where we
  just have to read consecutive bytes until we reach a \0.

  If no of these types is unambigously identified, we abort.


  Format of the gen_data files should be:
  ASCII|BINARY
  KEYWORD
  DOUBLE|FLOAT
  <SIZE>
  d1
  d2
  d3
  d4
  d5
  ....
  ....

  I.e. to write an ascii file (with C):

     fprintf(stream, "ASCII\n");
     fprintf(stream,"%s\n"keyword);
     fprintf(stream,"DOUBLE\n");
     fprintf(stream,"%d\n",elements);
     for (i = 0; i < elements; i++) {

     }  

  When written in binary with C you *MUST INCLUDE* the terminating \0
  at the end of the "BINARY", "KEYWORD" and "DOUBLE|FLOAT" strings.
     
  When written in binary with Fortran it is not necessary to include
  any form of termination, but it is essential that the write
  statement only writes the required number of bytes, with no extra
  spaces at the end.
*/




void gen_common_get_file_type(const char * filename , gen_data_file_type * _file_type , bool * fortran_endian_flip) {
  gen_data_file_type file_type;
  char buffer[32];
  if (fortio_is_fortran_file(filename , fortran_endian_flip)) {
    int record_length;
    fortio_type * fortio = fortio_fopen(filename , "r" , *fortran_endian_flip);
    record_length = fortio_fread_record(fortio , buffer);
    if (record_length != 6) 
      util_abort("%s: could not locate \'BINARY\' header in %s.\n",__func__ , filename);
    buffer[6] = '\0';
    util_strupr(buffer);
    if (strcmp(buffer , "BINARY") != 0)
      util_abort("%s: could not locate \'BINARY\' header in %s.\n",__func__ , filename);
    
    fortio_fclose(fortio);
    file_type = binary_fortran_file;
  } else {
    FILE * stream = util_fopen(filename , "r");
    long int init_pos = ftell(stream);
    util_fread(buffer , 1 , 5 , stream , __func__);
    buffer[5] = '\0';
    util_strupr(buffer);
    if (strcmp(buffer , "ASCII") == 0)
      file_type = ascii_file;
    else {
      fseek(stream , init_pos , SEEK_SET);
      util_fread(buffer , 1 , 7 , stream , __func__);
      if (buffer[6] != '\0') 
	util_abort("%s: the \"BINARY\" string identifier has not been \0 terminated - all header string must be \0 terminated\n",__func__);

      util_strupr(buffer);
      if (strcmp(buffer , "BINARY") == 0) 
	file_type = binary_C_file;
      else {
	util_abort("%s: could not determine BINARY / ASCII status of file:%s. Header: %s not recognized \n",__func__ , filename , buffer);
	file_type = binary_C_file; /* Dummy */
      }
    }
    fclose(stream);
  }
  *_file_type = file_type;
}


static void gen_common_fload_ascii_header(FILE * stream , const char * config_tag , char ** _file_tag , int* size, ecl_type_enum * ecl_type) {
  util_fskip_lines(stream , 1); /* We know the first line contains "ASCII". */
  {
    char * file_tag;
    file_tag = util_fscanf_alloc_token(stream);
    if (file_tag == NULL)
      util_abort("%s: could not locate tag. \n" , __func__);
    
    util_fskip_lines(stream , 1);
    *_file_tag = file_tag;
  }
  
  {
    char * string_type;
    string_type = util_fscanf_alloc_token(stream);
    util_strupr(string_type);
    if (strcmp(string_type, "DOUBLE") == 0)
      *ecl_type = ecl_double_type;
    else if (strcmp(string_type, "FLOAT") == 0)
      *ecl_type = ecl_float_type;
    else 
      util_abort("%s: type identiefier:%s  not recognized - valid values are FLOAT | DOUBLE \n",__func__ , string_type);
    free(string_type);
    util_fskip_lines(stream , 1);
  }

  if (!util_fscanf_int(stream , size))
    util_abort("%s: Failed to read the number of elements when parsing general data file. \n",__func__);
}


static void gen_common_fload_binary_C_header(FILE * stream , const char * config_tag , char ** _file_tag , int* size, ecl_type_enum * ecl_type) {
  util_exit("%s: not implemented yet ... \n");
}


static void gen_common_fload_binary_fortran_header(FILE * stream , const char * config_tag , char ** _file_tag , int* size, ecl_type_enum * ecl_type) {
  util_exit("%s: not implemented yet ... \n");
}



void gen_common_fload_header(gen_data_file_type file_type , FILE * stream , const char * config_tag , char ** file_tag , int * size, ecl_type_enum * ecl_type) {
  switch (file_type) {
  case(ascii_file):
    gen_common_fload_ascii_header(stream , config_tag , file_tag , size , ecl_type);
    break;
  case(binary_C_file):
    gen_common_fload_binary_C_header(stream , config_tag , file_tag , size , ecl_type);
    break;
  case(binary_fortran_file):
    gen_common_fload_binary_fortran_header(stream , config_tag , file_tag , size , ecl_type);
    break;
  default:
    util_abort("%s: internal error - invalid value in switch statement. \n",__func__);
  }

  /*
    Checking that the tags agree.
  */
  
  if (config_tag != NULL)
    if (strcmp(*file_tag , config_tag) != 0) 
      util_abort("%s: tags did not match: Config:%s  CurrentFile:%s \n",__func__ , config_tag , *file_tag);

}



static void gen_common_fload_binary_C_data(FILE * stream ,const char * src_file , gen_data_file_type file_type , ecl_type_enum ecl_type , int size, void * data) {
  util_exit("%s: not implemented yet ... \n");
}


static void gen_common_fload_binary_fortran_data(FILE * stream , const char * src_file , gen_data_file_type file_type , ecl_type_enum ecl_type , int size, void * data) {
  util_exit("%s: not implemented yet ... \n");
}


static void gen_common_fload_ascii_data(FILE * stream , const char * src_file , gen_data_file_type file_type , ecl_type_enum ecl_type , int size, void * _data) {
  int i;
  switch (ecl_type) {
  case(ecl_float_type):
    {
      float * data = (float *) _data;
      for (i=0; i < size; i++)
	if (fscanf(stream,"%g",&data[i]) != 1)
	  util_abort("%s: failed to read element %d of %d from %s. \n",__func__ , (i+1), size , src_file);
    }
    break;
  case(ecl_double_type):
    {
      double * data = (double *) _data;
      for (i=0; i < size; i++)
	if (fscanf(stream,"%lg",&data[i]) != 1)
	  util_abort("%s: failed to read element %d og %d from %s. \n",__func__ , (i+1), size , src_file);
    }
    break;
  case(ecl_int_type):
    {
      int * data = (int *) _data;
      for (i=0; i < size; i++) 
	if (fscanf(stream,"%d" , &data[i]) != 1)
	  util_abort("%s: failed to read element %d og %d from %s. \n",__func__ , (i+1), size , src_file);
    }
    break;
  default:
    util_abort("%s: unrecognized/not supported data_type:%d.\n",__func__ , ecl_type);
  }
}



void gen_common_fload_data(FILE * stream , const char * src_file , gen_data_file_type file_type , ecl_type_enum ecl_type , int size, void * data) {
  switch (file_type) {
  case(ascii_file):
    gen_common_fload_ascii_data(stream , src_file , file_type , ecl_type , size , data);
    break;
  case(binary_C_file):
    gen_common_fload_binary_C_data(stream , src_file , file_type , ecl_type , size , data);
    break;
  case(binary_fortran_file):
    gen_common_fload_binary_fortran_data(stream , src_file , file_type , ecl_type , size , data);
    break;
  default:
    util_abort("%s: internal error - invalid value in switch statement. \n",__func__);
  }
}




static void gen_common_fskip_binary_C_data(FILE * stream ,const char * src_file , gen_data_file_type file_type , ecl_type_enum ecl_type , int size) {
  util_exit("%s: not implemented yet ... \n");
}


static void gen_common_fskip_binary_fortran_data(FILE * stream , const char * src_file , gen_data_file_type file_type , ecl_type_enum ecl_type , int size) {
  util_exit("%s: not implemented yet ... \n");
}


static void gen_common_fskip_ascii_data(FILE * stream , const char * src_file , gen_data_file_type file_type , ecl_type_enum ecl_type , int size) {
  int i;
  switch (ecl_type) {
  case(ecl_float_type):
    {
      float  data;
      for (i=0; i < size; i++)
	if (fscanf(stream,"%g",&data) != 1)
	  util_abort("%s: failed to read element %d of %d from %s. \n",__func__ , (i+1), size , src_file);
    }
    break;
  case(ecl_double_type):
    {
      double data;
      for (i=0; i < size; i++)
	if (fscanf(stream,"%lg",&data) != 1)
	  util_abort("%s: failed to read element %d og %d from %s. \n",__func__ , (i+1), size , src_file);
    }
    break;
  case(ecl_int_type):
    {
      int data;
      for (i=0; i < size; i++)
	if (fscanf(stream,"%d",&data) != 1)
	  util_abort("%s: failed to read element %d og %d from %s. \n",__func__ , (i+1), size , src_file);
    }
    break;
  default:
    util_abort("%s: unrecognized/not supported data_type:%d.\n",__func__ , ecl_type);
  }
}



void gen_common_fskip_data(FILE * stream , const char * src_file , gen_data_file_type file_type , ecl_type_enum ecl_type , int size) {
  switch (file_type) {
  case(ascii_file):
    gen_common_fskip_ascii_data(stream , src_file , file_type , ecl_type , size);
    break;
  case(binary_C_file):
    gen_common_fskip_binary_C_data(stream , src_file , file_type , ecl_type , size);
    break;
  case(binary_fortran_file):
    gen_common_fskip_binary_fortran_data(stream , src_file , file_type , ecl_type , size);
    break;
  default:
    util_abort("%s: internal error - invalid value in switch statement. \n",__func__);
  }
}



double gen_common_iget_double(int index , int size , ecl_type_enum ecl_type , void * _data) {
  if (index < 0 || index >= size) {
    util_abort("%s: asked for element:%d - only has:%d elements - aborting \n",__func__ , index , size);
    return 0; /* Dummy */
  }

  if (ecl_type == ecl_double_type) {
    const double * data = (const double * ) _data;
    return data[index];
  } else if (ecl_type == ecl_float_type) {
    const float * data = (const float * ) _data;
    return (double ) data[index];
  } else {
    util_abort("%s: internal error: ecl_type:%d not supported.\n",__func__ , ecl_type);
    return 0; /* Dummy */
  }
}
