#ifndef __GEN_COMMON_H__
#define __GEN_COMMON_H__
#include <ecl_util.h>
#include <stdlib.h>
#include <stdio.h>

void    * gen_common_fscanf_alloc(const char * , ecl_type_enum  , int * );
void    * gen_common_fread_alloc(const char *  , ecl_type_enum   , int * );
void    * gen_common_fload_alloc(const char *  , gen_data_file_format_type , ecl_type_enum   , ecl_type_enum * , int * );

#endif
