/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'analysis_module.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <string.h>
#include <stdbool.h>
#include <stdio.h>
#include <dlfcn.h>
#include <matrix.h>
#include <util.h>

#include <analysis_module.h>
#include <analysis_table.h>

#define ANALYSIS_MODULE_TYPE_ID 6610123

struct analysis_module_struct {
  UTIL_TYPE_ID_DECLARATION;
  void                       * lib_handle;
  void                       * module_data;

  analysis_free_ftype        * freef;
  analysis_alloc_ftype       * alloc;
  analysis_initX_ftype       * initX;
    
  analysis_set_int_ftype     * set_int;
  analysis_set_double_ftype  * set_double;
  analysis_set_string_ftype  * set_string;
};



static analysis_module_type * analysis_module_alloc_empty( ) {
  analysis_module_type * module = util_malloc( sizeof * module , __func__ );
  UTIL_TYPE_ID_INIT( module , ANALYSIS_MODULE_TYPE_ID );

  module->lib_handle  = NULL;
  module->initX       = NULL;
  module->set_int     = NULL;
  module->set_double  = NULL;
  module->set_string  = NULL;
  module->module_data = NULL;

  return module;
}


static analysis_module_type * analysis_module_alloc__( const analysis_table_type * analysis , void * lib_handle ) {
  analysis_module_type * module = analysis_module_alloc_empty();
  
  module->lib_handle = lib_handle;
  module->initX      = analysis->initX;
  module->set_int    = analysis->set_int;
  module->set_double = analysis->set_double;
  module->set_string = analysis->set_string;
  module->alloc      = analysis->alloc;
  module->freef      = analysis->freef;
  
  if (module->alloc != NULL)
    module->module_data = module->alloc( );
  
  return module;
}





static analysis_module_type * analysis_module_alloc( const char * libname , const char * table_name , const char * name ) {
  analysis_module_type * module = NULL;
  void * lib_handle = dlopen( NULL , RTLD_NOW );
  if (lib_handle != NULL) {
    analysis_table_type * analysis_table = (analysis_table_type *) dlsym( lib_handle , table_name );
    if (analysis_table != NULL) {
      analysis_table_type * analysis  = analysis_table;
      if (name == NULL) 
        /* No name specified - load the first module. */
        
        module = analysis_module_alloc__( analysis , lib_handle );
      else {
        while (true) {

          if (analysis->name == NULL)
            break;
        
          if (util_string_equal( analysis->name , name)) {
            module = analysis_module_alloc__( analysis , lib_handle );
            break;
          }
          
          analysis++;
        }
      }
    } else
      fprintf(stderr , "Failed to load symbol table:%s. Error:%s \n",table_name , dlerror());
    
    if (module == NULL) 
      dlclose( lib_handle );
  } else 
    fprintf(stderr , "Failed to load library:%s. Error:%s \n",libname , dlerror());
  
  return module;
}

analysis_module_type * analysis_module_alloc_internal( const char * symbol_table , const char * module_name ) {
  return analysis_module_alloc( NULL , symbol_table , module_name );
}


analysis_module_type * analysis_module_alloc_external( const char * lib_name , const char * name ) {
  return analysis_module_alloc( lib_name , EXTERNAL_MODULE_TABLE , name );
}



/*****************************************************************/

static UTIL_SAFE_CAST_FUNCTION( analysis_module , ANALYSIS_MODULE_TYPE_ID )


void analysis_module_free( analysis_module_type * module ) {
  if (module->freef != NULL)
    module->freef( module->module_data );
  
  dlclose( module->lib_handle );
  free( module );
}


void analysis_module_free__( void * arg) {
  analysis_module_type * module = analysis_module_safe_cast( arg );
  analysis_module_free( module );
}


void analysis_module_initX(analysis_module_type * module , 
                           matrix_type * X , 
                           matrix_type * S , 
                           matrix_type * R , 
                           matrix_type * innov , 
                           matrix_type * E , 
                           matrix_type * D) {
  module->initX(module->module_data , X , S , R , innov , E , D );
}

bool analysis_module_needs_ED( const analysis_module_type * module ) {
  return true;
}

/*****************************************************************/

static bool analysis_module_set_int(analysis_module_type * module , const char * flag , int value) {
  if (module->set_int != NULL) 
    return module->set_int( module->module_data , flag , value );
  else
    return false;
}

static bool analysis_module_set_double(analysis_module_type * module , const char * var , double value) {
  if (module->set_double != NULL)
    return module->set_double( module->module_data , var , value );
  else
    return false;
}

static bool analysis_module_set_string(analysis_module_type * module , const char * var , const char * value) {
  if (module->set_string != NULL)
    return module->set_string( module->module_data , var , value );
  else
    return false;
}

/* 
   The same variable name can NOT be used for several variables.
*/

bool analysis_module_set_var( analysis_module_type * module , const char * var_name , const char * string_value ) {
  bool set_ok = false;
  {
    int  int_value;
    
    if (util_sscanf_int( string_value , &int_value )) 
      set_ok = analysis_module_set_int( module , var_name , int_value );
    
    if (set_ok)
      return true;
  }
  
  {
    double double_value;
    if (util_sscanf_double( string_value , &double_value )) 
      set_ok = analysis_module_set_double( module , var_name , double_value );

    if (set_ok)
      return true;
  }

  return analysis_module_set_string( module , var_name , string_value );
}
