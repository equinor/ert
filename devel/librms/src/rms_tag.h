#ifndef __RMS_TAG_H__
#define __RMS_TAG_H__
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <hash.h>
#include <rms_tagkey.h>



typedef struct rms_tag_struct rms_tag_type;

int               rms_tag_get_datakey_sizeof_ctype(const rms_tag_type * );
const char      * rms_tag_get_namekey_name(const rms_tag_type * );
const char      * rms_tag_get_name(const rms_tag_type *);
rms_tagkey_type * rms_tag_get_datakey(const rms_tag_type *);
void           	  rms_tag_free(rms_tag_type *);
rms_tag_type    * rms_tag_fread_alloc(FILE *, hash_type *, bool , bool *);
bool           	  rms_tag_name_eq(const rms_tag_type *, const char * , const char *, const char *);
rms_tagkey_type * rms_tag_get_key(const rms_tag_type *, const char *);
void       	  rms_tag_fwrite_filedata(const char * , FILE *stream);
void       	  rms_tag_fwrite_eof(FILE *stream);
void              rms_tag_fwrite(const rms_tag_type * , FILE * );
void              rms_tag_printf(const rms_tag_type * , FILE * );
const char      * rms_tag_name_ref(const rms_tag_type * );
rms_tag_type    * rms_tag_alloc_dimensions(int  , int , int );
void              rms_tag_fwrite_dimensions(int , int , int  , FILE *);
void              rms_tag_fwrite_parameter(const char *, const rms_tagkey_type *, FILE *);
#endif
