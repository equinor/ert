#ifndef __GEN_COMMON_H__
#define __GEN_COMMON_H__
#include <ecl_util.h>
#include <stdlib.h>
#include <stdio.h>

typedef enum {ascii_file , binary_C_file , binary_fortran_file} gen_data_file_type;

void          gen_common_get_file_type(const char * , gen_data_file_type * , bool * );
void          gen_common_fload_header(gen_data_file_type , FILE * , const char *, char ** , int *, ecl_type_enum *);
void          gen_common_fload_data(FILE * , const char * , gen_data_file_type , ecl_type_enum , int , void * );
void          gen_common_fskip_data(FILE * , const char *  , gen_data_file_type , ecl_type_enum , int);

#endif
