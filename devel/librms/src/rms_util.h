#ifndef __RMS_UTIL_H__
#define __RMS_UTIL_H__
#include <stdio.h>
#include <stdlib.h>

void   rms_util_translate_undef(void * , int , int , const void * , const void * );
void   rms_util_set_fortran_data(void *, const void * , int , int , int  , int);
void   rms_util_read_fortran_data(const void *, void * , int , int , int , int);
void   rms_util_fskip_string(FILE *);
int    rms_util_fread_strlen(FILE *);
bool   rms_util_fread_string(char * ,  int , FILE *);
void   rms_util_fwrite_string(const char * string , FILE *stream);
void   rms_util_fwrite_comment(const char *  , FILE *);
void   rms_util_fwrite_newline(FILE *stream);

#endif
