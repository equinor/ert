/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'multflt_config.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#ifndef __MULTFLT_CONFIG_H__
#define __MULTFLT_CONFIG_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <scalar_config.h>
#include <scalar.h>

typedef struct multflt_config_struct multflt_config_type;


scalar_config_type  * multflt_config_get_scalar_config( const multflt_config_type * );
int                   multflt_config_get_index(const multflt_config_type *  , const char * );
multflt_config_type * multflt_config_fscanf_alloc(const char *);
void                  multflt_config_free(multflt_config_type *);
void                  multflt_config_transform(const multflt_config_type * , const double * , double *);
void                  multflt_config_truncate(const multflt_config_type * , scalar_type * );
int                   multflt_config_get_data_size(const multflt_config_type * );
const char          * multflt_config_get_name(const multflt_config_type * , int );

SAFE_CAST_HEADER(multflt_config);
VOID_FREE_HEADER(multflt_config);
VOID_GET_DATA_SIZE_HEADER(multflt);
#ifdef __cplusplus
}
#endif
#endif
