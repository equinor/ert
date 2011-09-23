/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'analysis_module.h' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#ifndef __ANALYSIS_MODULE_H__
#define __ANALYSIS_MODULE_H__
#ifdef  __cplusplus
extern "C" {
#endif

#include <matrix.h>

typedef struct analysis_module_struct analysis_module_type;


analysis_module_type * analysis_module_alloc_external( const char * user_name , const char * libname );
analysis_module_type * analysis_module_alloc_internal( const char * user_name , const char * symbol_table );
void                   analysis_module_free( analysis_module_type * module );
void                   analysis_module_free__( void * arg);

void analysis_module_initX(analysis_module_type * module , 
                           matrix_type * X , 
                           matrix_type * S , 
                           matrix_type * R , 
                           matrix_type * innov , 
                           matrix_type * E , 
                           matrix_type * D ,
                           matrix_type * randrot);

bool  analysis_module_set_var( analysis_module_type * module , const char * var_name , const char * string_value );

bool                   analysis_module_get_option( const analysis_module_type * module , long flag);
void                   analysis_module_complete_update( analysis_module_type * module );
void                   analysis_module_init_update( analysis_module_type * module , 
                                                    matrix_type * S , 
                                                    matrix_type * R , 
                                                    matrix_type * innov , 
                                                    matrix_type * E , 
                                                    matrix_type * D );

#ifdef  __cplusplus
}
#endif
#endif
