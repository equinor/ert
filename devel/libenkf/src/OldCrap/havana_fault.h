/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'havana_fault.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#ifndef __HAVANA_H__
#define __HAVANA_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <havana_fault_config.h>
#include <enkf_util.h>
#include <enkf_macros.h>

typedef struct havana_fault_struct havana_fault_type;


void             havana_fault_output_transform(const havana_fault_type * );
void             havana_fault_get_output_data(const havana_fault_type * , double * );
const double   * havana_fault_get_output_ref(const havana_fault_type * );
const double   * havana_fault_get_data_ref(const havana_fault_type * );
void             havana_fault_get_data(const havana_fault_type * , double * );
void             havana_fault_set_data(havana_fault_type * , const double * );

havana_fault_type    * havana_fault_alloc(const havana_fault_config_type * );

void             havana_fault_free(havana_fault_type *);
/*void             havana_fault_ecl_write(const havana_fault_type * , const char *);*/
void             havana_fault_ens_write(const havana_fault_type * , const char *);
void             havana_fault_ens_read(havana_fault_type * , const char *);
void             havana_fault_truncate(havana_fault_type * );
havana_fault_type   *  havana_fault_alloc_mean(int , const havana_fault_type **);
const char     * havana_fault_get_name(const havana_fault_type * , int );
void             havana_fault_export(const havana_fault_type * , int * , char ***, double **);
void             havana_fault_upgrade_103(const char * filename);


SAFE_CAST_HEADER(havana_fault)
ALLOC_STATS_HEADER(havana_fault)
VOID_USER_GET_HEADER(havana_fault)
VOID_ECL_WRITE_HEADER  (havana_fault)
VOID_COPYC_HEADER      (havana_fault);
VOID_SERIALIZE_HEADER  (havana_fault);
VOID_DESERIALIZE_HEADER  (havana_fault);
VOID_FREE_DATA_HEADER(havana_fault)
VOID_INITIALIZE_HEADER(havana_fault);
VOID_FREE_HEADER       (havana_fault);
MATH_OPS_VOID_HEADER(havana_fault);
VOID_ALLOC_HEADER(havana_fault);
VOID_ECL_WRITE_HEADER(havana_fault);
VOID_REALLOC_DATA_HEADER(havana_fault);
VOID_LOAD_HEADER(havana_fault)
VOID_STORE_HEADER(havana_fault)
VOID_MATRIX_SERIALIZE_HEADER(havana_fault)
VOID_MATRIX_DESERIALIZE_HEADER(havana_fault)
#ifdef __cplusplus
}
#endif
#endif
