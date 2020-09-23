/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'rms_file.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <stdbool.h>

#include <ert/util/hash.hpp>
#include <ert/util/vector.hpp>
#include <ert/util/util.hpp>

#include <ert/rms/rms_type.hpp>
#include <ert/rms/rms_util.hpp>
#include <ert/rms/rms_tag.hpp>
#include <ert/rms/rms_file.hpp>
#include <ert/rms/rms_tagkey.hpp>

#include <ert/ecl/ecl_kw.h>
#include <ert/ecl/ecl_type.h>

/*****************************************************************/
static const char * rms_ascii_header      = "roff-asc";
static const char * rms_binary_header     = "roff-bin";

static const char * rms_comment1          = "ROFF file";
static const char * rms_comment2          = "Creator: RMS - Reservoir Modelling System, version 8.1";


struct rms_file_struct {
  char         * filename;
  bool           endian_convert;
  bool           fmt_file;
  hash_type    * type_map;
  vector_type  * tag_list;
  FILE         * stream;
};



/*****************************************************************/
/* Pure roff routines */

static bool rms_fmt_file(const rms_file_type *rms_file) {
  char filetype[9];
  rms_util_fread_string( filetype , 9 , rms_file->stream);

  if (strncmp(filetype , rms_binary_header , 8) == 0)
    return false;
  if (strncmp(filetype , rms_ascii_header , 8) == 0)
    return true;

  fprintf(stderr,"%s: header : %8s not recognized in file: %s - aborting \n",__func__ , filetype , rms_file->filename);
  abort();
  return false; // will not happen
}


static void rms_file_add_tag(rms_file_type *rms_file , const rms_tag_type *tag) {
  vector_append_owned_ref(rms_file->tag_list , tag , rms_tag_free__ );
}


rms_tag_type * rms_file_get_tag_ref(const rms_file_type *rms_file ,
                                    const char *tagname ,
                                    const char *keyname ,
                                    const char *keyvalue, bool abort_on_error) {

  rms_tag_type *return_tag = NULL;

  int size = vector_get_size( rms_file->tag_list );
  for (int index = 0; index < size; ++index) {
    rms_tag_type *tag = (rms_tag_type*)vector_iget( rms_file->tag_list , index );
    if (rms_tag_name_eq(tag , tagname , keyname , keyvalue)) {
      return_tag = tag;
      break;
    }
  }

  if (return_tag == NULL && abort_on_error) {
    if (keyname != NULL && keyvalue != NULL)
      fprintf(stderr,"%s: failed to find tag:%s with key:%s=%s in file:%s - aborting \n",__func__ , tagname , keyname , keyvalue , rms_file->filename);
    else
      fprintf(stderr,"%s: failed to find tag:%s in file:%s - aborting \n",__func__ , tagname , rms_file->filename);
  }
  return return_tag;
}


/**
    This function allocates and rms_file_type * handle, but it does
    not load the file content.
*/
rms_file_type * rms_file_alloc(const char *filename, bool fmt_file) {
  rms_file_type *rms_file   = (rms_file_type*)malloc(sizeof *rms_file);
  rms_file->endian_convert  = false;
  rms_file->type_map        = hash_alloc();
  rms_file->tag_list        = vector_alloc_new();

  hash_insert_hash_owned_ref(rms_file->type_map , "byte"   , rms_type_alloc(rms_byte_type ,    1) ,  rms_type_free);
  hash_insert_hash_owned_ref(rms_file->type_map , "bool"   , rms_type_alloc(rms_bool_type,     1) ,  rms_type_free);
  hash_insert_hash_owned_ref(rms_file->type_map , "int"    , rms_type_alloc(rms_int_type ,     4) ,  rms_type_free);
  hash_insert_hash_owned_ref(rms_file->type_map , "float"  , rms_type_alloc(rms_float_type  ,  4) ,  rms_type_free);
  hash_insert_hash_owned_ref(rms_file->type_map , "double" , rms_type_alloc(rms_double_type ,  8) ,  rms_type_free);

  hash_insert_hash_owned_ref(rms_file->type_map , "char"   , rms_type_alloc(rms_char_type   , -1) ,  rms_type_free);   /* Char are a f*** mix of vector and scalar */

  rms_file->filename = NULL;
  rms_file->stream   = NULL;
  rms_file_set_filename(rms_file , filename , fmt_file);
  return rms_file;
}


void rms_file_set_filename(rms_file_type * rms_file , const char *filename , bool fmt_file) {
  rms_file->filename = util_realloc_string_copy(rms_file->filename , filename);
  rms_file->fmt_file   = fmt_file;
}


void rms_file_free_data(rms_file_type * rms_file) {
  vector_clear( rms_file->tag_list );
}


void rms_file_free(rms_file_type * rms_file) {
  rms_file_free_data(rms_file);
  vector_free( rms_file->tag_list );
  hash_free(rms_file->type_map);
  free(rms_file->filename);
  free(rms_file);
}


static int rms_file_get_dim(const rms_tag_type *tag , const char *dim_name) {
  rms_tagkey_type *key = rms_tag_get_key(tag , dim_name);
  if (key == NULL) {
    fprintf(stderr,"%s: failed to find tagkey:%s aborting \n" , __func__ , dim_name);
    abort();
  }
  return * (int *) rms_tagkey_get_data_ref(key);
}


rms_tag_type * rms_file_get_dim_tag_ref(const rms_file_type * rms_file) {
  return rms_file_get_tag_ref(rms_file , "dimensions" , NULL , NULL , true);
}


void rms_file_get_dims(const rms_file_type * rms_file , int * dims) {
  rms_tag_type *tag = rms_file_get_dim_tag_ref(rms_file);
  dims[0] = rms_file_get_dim(tag , "nX");
  dims[1] = rms_file_get_dim(tag , "nY");
  dims[2] = rms_file_get_dim(tag , "nZ");
}


FILE * rms_file_get_FILE(const rms_file_type * rms_file) {
  return rms_file->stream;
}


static void rms_file_init_fread(rms_file_type * rms_file) {

  rms_file->fmt_file = rms_fmt_file( rms_file );
  if (rms_file->fmt_file) {
    fprintf(stderr,"%s only binary files implemented - aborting \n",__func__);
    abort();
  }
  /* Skipping two comment lines ... */
  rms_util_fskip_string(rms_file->stream);
  rms_util_fskip_string(rms_file->stream);
  {
    bool eof_tag;
    rms_tag_type    * filedata_tag = rms_tag_fread_alloc(rms_file->stream,
                                                         rms_file->type_map,
                                                         rms_file->endian_convert,
                                                         &eof_tag);
    rms_tagkey_type * byteswap_key = rms_tag_get_key(filedata_tag , "byteswaptest");
    if (byteswap_key == NULL) {
      fprintf(stderr,"%s: failed to find filedata/byteswaptest - aborting \n", __func__);
      abort();
    }
    int byteswap_value = *( int *) rms_tagkey_get_data_ref(byteswap_key);
    if (byteswap_value == 1)
      rms_file->endian_convert = false;
    else
      rms_file->endian_convert = true;
    rms_tag_free(filedata_tag);
  }
}


rms_tag_type * rms_file_fread_alloc_tag(rms_file_type * rms_file,
                                        const char *tagname,
                                        const char *keyname,
                                        const char *keyvalue) {
  rms_tag_type * tag = NULL;
  rms_file_fopen_r(rms_file);

  long int start_pos = util_ftell(rms_file->stream);
  fseek(rms_file->stream , 0 , SEEK_SET);
  rms_file_init_fread(rms_file);
  while (true) {
    bool eof_tag = false;  // will be set by rms_tag
    rms_tag_type * tmp_tag = rms_tag_fread_alloc(rms_file->stream,
                                                 rms_file->type_map,
                                                 rms_file->endian_convert,
                                                 &eof_tag);

    if (!rms_tag_name_eq(tmp_tag, tagname, keyname, keyvalue)) {
      rms_tag_free(tmp_tag);
      continue;
    }
    tag = tmp_tag;
    break;
  }

  if (tag == NULL) {
    fseek(rms_file->stream , start_pos , SEEK_SET);
    util_abort("%s: could not find tag: \"%s\" (with %s=%s) in file:%s - aborting.\n",
               __func__,
               tagname,
               keyname,
               keyvalue,
               rms_file->filename);
  }

  rms_file_fclose(rms_file);
  return tag;
}


FILE * rms_file_fopen_r(rms_file_type *rms_file) {
  rms_file->stream = util_fopen(rms_file->filename , "r");
  return rms_file->stream;
}


FILE * rms_file_fopen_w(rms_file_type *rms_file) {
  rms_file->stream = util_mkdir_fopen(rms_file->filename , "w");
  return rms_file->stream;
}

void rms_file_fclose(rms_file_type * rms_file) {
  fclose(rms_file->stream);
  rms_file->stream = NULL;
}


rms_tagkey_type * rms_file_fread_alloc_data_tagkey(rms_file_type * rms_file,
                                                   const char *tagname,
                                                   const char *keyname,
                                                   const char *keyvalue) {
  rms_tag_type * tag = rms_file_fread_alloc_tag(rms_file,
                                                tagname,
                                                keyname,
                                                keyvalue);
  if (tag == NULL)
    return NULL;

  rms_tagkey_type *tagkey = rms_tagkey_copyc(rms_tag_get_key(tag, "data"));
  rms_tag_free(tag);
  return tagkey;
}


/*static */
void rms_file_init_fwrite(const rms_file_type * rms_file , const char * filetype) {
  if (!rms_file->fmt_file)
    rms_util_fwrite_string(rms_binary_header , rms_file->stream);
  else {
    fprintf(stderr,"%s: Sorry only binary writes implemented ... \n",__func__);
    rms_util_fwrite_string(rms_ascii_header , rms_file->stream);
  }

  rms_util_fwrite_comment(rms_comment1 , rms_file->stream);
  rms_util_fwrite_comment(rms_comment2 , rms_file->stream);
  rms_tag_fwrite_filedata(filetype , rms_file->stream);
}


void rms_file_complete_fwrite(const rms_file_type * rms_file) {
  rms_tag_fwrite_eof(rms_file->stream);
}


bool rms_file_is_roff(FILE * stream) {
  const int len              = strlen(rms_comment1);
  char *header               = (char*)malloc(strlen(rms_comment1) + 1);
  const long int current_pos = util_ftell(stream);
  bool roff_file             = false;

  /* Skipping #roff-bin#0#  WILL Fail with formatted files */
  fseek(stream, 1 + 1 + 8, SEEK_CUR);


  rms_util_fread_string(header , len+1 , stream);
  if (strncmp(rms_comment1 , header , len) == 0)
    roff_file = true;

  fseek(stream , current_pos , SEEK_SET);
  free(header);
  return roff_file;
}
