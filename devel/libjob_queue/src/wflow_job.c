/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'wflow_job.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <stdbool.h>
#include <stdlib.h>
#include <dlfcn.h>

#include <int_vector.h>
#include <util.h>
#include <type_macros.h>

#include <config.h>

#include <wflow_job.h>


/* The default values are interepreted as no limit. */
#define DEFAULT_INTERNAL false


#define MIN_ARG_KEY    "MIN_ARG"
#define MAX_ARG_KEY    "MAX_ARG"
#define ARG_TYPE_KEY   "ARG_TYPE"
#define INTERNAL_KEY   "INTERNAL"
#define MODULE_KEY     "MODULE" 
#define FUNCTION_KEY   "FUNCTION"
#define EXECUTABLE_KEY "EXECUTABLE"

#define NULL_STRING         "NULL"
#define WFLOW_JOB_STRING_TYPE "STRING"
#define WFLOW_JOB_INT_TYPE    "INT"
#define WFLOW_JOB_FLOAT_TYPE  "FLOAT"

#define WFLOW_JOB_TYPE_ID 614441


struct wflow_job_struct {
  UTIL_TYPE_ID_DECLARATION;
  bool              internal;
  int               min_arg;
  int               max_arg;   
  int_vector_type * arg_types;     // Should contain values from the config_item_types enum in config.h.
  char            * executable;
  char            * module;
  char            * function;
  char            * name;
  void            * lib_handle;
  void            * dl_func;
  bool              valid;
};


bool wflow_job_internal( const wflow_job_type * wflow_job ) {
  return wflow_job->internal;
}

const char * wflow_job_get_name( const wflow_job_type * wflow_job ) {
  return wflow_job->name;
}


config_type * wflow_job_alloc_config() {
  config_type * config = config_alloc();
  {
    config_schema_item_type * item;
  
    item = config_add_schema_item( config , MIN_ARG_KEY , false );
    config_schema_item_set_argc_minmax( item , 1 , 1 , 1 , (const config_item_types[1]) {CONFIG_INT});

    item = config_add_schema_item( config , MAX_ARG_KEY , false );
    config_schema_item_set_argc_minmax( item , 1 , 1 , 1 , (const config_item_types[1]) {CONFIG_INT});

    item = config_add_schema_item( config , ARG_TYPE_KEY , false );
    config_schema_item_set_argc_minmax( item , 2 , 2 , 2 , (const config_item_types[2]) {CONFIG_INT , CONFIG_STRING});
    config_schema_item_set_indexed_selection_set( item , 1 , 3 , (const char *[3]) {WFLOW_JOB_STRING_TYPE , WFLOW_JOB_INT_TYPE , WFLOW_JOB_FLOAT_TYPE});

    /*****************************************************************/
    item = config_add_schema_item( config , EXECUTABLE_KEY , false );
    config_schema_item_set_argc_minmax( item , 1 , 1 , 1 , (const config_item_types[1]) {CONFIG_EXECUTABLE});

    /*---------------------------------------------------------------*/
    
    item = config_add_schema_item( config , FUNCTION_KEY , false );
    config_schema_item_set_argc_minmax( item , 1 , 1 , 1 , (const config_item_types[1]) {CONFIG_STRING});

    item = config_add_schema_item( config , MODULE_KEY , false );
    config_schema_item_set_argc_minmax( item , 1 , 1 , 1 , (const config_item_types[1]) {CONFIG_STRING});
    /*****************************************************************/

    item = config_add_schema_item( config , INTERNAL_KEY , false );
    config_schema_item_set_argc_minmax( item , 1 , 1 , 1 , (const config_item_types[1]) {CONFIG_BOOLEAN});    
    
  }
  return config;
}



static UTIL_SAFE_CAST_FUNCTION(wflow_job , WFLOW_JOB_TYPE_ID );

void wflow_job_update_config_compiler( const wflow_job_type * wflow_job , config_type * config_compiler ) {
  config_schema_item_type * item = config_add_schema_item( config_compiler , wflow_job->name , false );
  /* 
     Ensure that the arg_types mapping is at least as large as the
     max_arg value. The arg_type vector will be left padded with
     CONFIG_STRING values.
  */
  {
    int argc_types = int_vector_size( wflow_job->arg_types );
    config_item_types * arg_types = (config_item_types *) int_vector_get_ptr( wflow_job->arg_types );
    
    config_schema_item_set_argc_minmax( item , wflow_job->min_arg , wflow_job->max_arg , argc_types , arg_types);
  }
}


wflow_job_type * wflow_job_alloc( const char * name , bool internal ) {
  wflow_job_type * wflow_job = util_malloc( sizeof * wflow_job );
  UTIL_TYPE_ID_INIT( wflow_job , WFLOW_JOB_TYPE_ID );
  wflow_job->internal   = internal;      // this can not be changed run-time.
  wflow_job->min_arg    = CONFIG_DEFAULT_ARG_MIN;
  wflow_job->max_arg    = CONFIG_DEFAULT_ARG_MAX;
  wflow_job->arg_types  = int_vector_alloc( 0 , CONFIG_STRING );

  wflow_job->executable = NULL;
  wflow_job->module     = NULL;
  wflow_job->function   = NULL;

  if (name == NULL)
    util_abort("%s: trying to create wflow_job with name == NULL - illegal\n",__func__);
  else
    wflow_job->name       = util_alloc_string_copy( name );

  wflow_job->valid      = false;

  return wflow_job;
}


void wflow_job_set_executable( wflow_job_type * wflow_job , const char * executable ) {
  wflow_job->executable = util_realloc_string_copy( wflow_job->executable , executable );
}


void wflow_job_set_module( wflow_job_type * wflow_job , const char * module) {
  if (strcmp(module  ,NULL_STRING) == 0)
    module = NULL;

  wflow_job->module = util_realloc_string_copy( wflow_job->module , module );
}


void wflow_job_set_function( wflow_job_type * wflow_job , const char * function) {
  wflow_job->function = util_realloc_string_copy( wflow_job->function , function );
}


void wflow_job_iset_argtype( wflow_job_type * wflow_job , int iarg , config_item_types type) {
  if (type == CONFIG_STRING || type == CONFIG_INT || type == CONFIG_FLOAT)
    int_vector_iset( wflow_job->arg_types , iarg , type );
}

void wflow_job_set_min_arg( wflow_job_type * wflow_job , int min_arg) {
  wflow_job->min_arg = min_arg;
}

void wflow_job_set_max_arg( wflow_job_type * wflow_job , int max_arg) {
  wflow_job->max_arg = max_arg;
}

int wflow_job_get_min_arg( const wflow_job_type * wflow_job ) {
  return wflow_job->min_arg;
}
 
int wflow_job_get_max_arg( wflow_job_type * wflow_job ) {
  return wflow_job->max_arg;
}



static void wflow_job_iset_argtype_string( wflow_job_type * wflow_job , int iarg , const char * arg_type) {
  config_item_types type = CONFIG_INVALID;

  if (strcmp( arg_type , WFLOW_JOB_STRING_TYPE) == 0)
    type = CONFIG_STRING;
  else if (strcmp( arg_type , WFLOW_JOB_INT_TYPE) == 0)
    type = CONFIG_INT;
  else if (strcmp( arg_type , WFLOW_JOB_FLOAT_TYPE) == 0)
    type = CONFIG_FLOAT;

  if (type != CONFIG_INVALID)
    wflow_job_iset_argtype( wflow_job , iarg , type );
  
}


static void wflow_job_validate( wflow_job_type * wflow_job ) {
  if (!wflow_job->internal) {
    if (wflow_job->executable != NULL) {
      if (util_is_executable( wflow_job->executable ) && 
          (wflow_job->module == wflow_job->function) && 
          (wflow_job->module == NULL))
        wflow_job->valid = true;
    }
  } else {
    if ((wflow_job->executable == NULL) && (wflow_job->function != NULL)) {
      wflow_job->lib_handle = dlopen( wflow_job->module , RTLD_NOW );
      if (wflow_job->lib_handle != NULL) {
        wflow_job->dl_func = dlsym( wflow_job->lib_handle , wflow_job->function );
        if (wflow_job->dl_func != NULL)
          wflow_job->valid = true;
        else 
          fprintf(stderr,"Failed to load symbol:%s Error:%s \n",wflow_job->function , dlerror());
      } else {
        if (wflow_job->module != NULL)
          fprintf(stderr,"Failed to load module:%s Error:%s \n",wflow_job->module , dlerror());
      }
    }
  }
}



wflow_job_type * wflow_job_config_alloc( const char * name , config_type * config , const char * config_file) {
  config_clear( config );
  if (config_parse( config , config_file , "--", NULL , NULL , true , true)) {
    
    bool internal = DEFAULT_INTERNAL;
    if (config_item_set( config , INTERNAL_KEY))
      internal = config_iget_as_bool( config , INTERNAL_KEY , 0 , 0 );
    
    {
      wflow_job_type * wflow_job = wflow_job_alloc( name , internal );
      
      if (config_item_set( config , MIN_ARG_KEY))
        wflow_job_set_min_arg( wflow_job , config_iget_as_int( config , MIN_ARG_KEY , 0 , 0 ));
      
      if (config_item_set( config , MAX_ARG_KEY))
        wflow_job_set_max_arg( wflow_job , config_iget_as_int( config , MAX_ARG_KEY , 0 , 0 ));
      
      {
        int i;
        for (i=0; i < config_get_occurences( config , ARG_TYPE_KEY); i++) {
          int iarg = config_iget_as_int( config , ARG_TYPE_KEY , i , 0 );
          const char * arg_type = config_iget( config , ARG_TYPE_KEY , i , 1 );
          
          wflow_job_iset_argtype_string( wflow_job , iarg , arg_type );
        }
      }
      
      if (config_item_set( config , MODULE_KEY))
        wflow_job_set_module( wflow_job , config_iget( config , MODULE_KEY , 0 , 0 ));
      
      if (config_item_set( config , FUNCTION_KEY))
        wflow_job_set_function( wflow_job , config_iget( config , FUNCTION_KEY , 0 , 0 ));
      
      if (config_item_set( config , EXECUTABLE_KEY))
        wflow_job_set_executable( wflow_job , config_iget( config , EXECUTABLE_KEY , 0 , 0 ));
      
      wflow_job_validate( wflow_job );
      
      if (!wflow_job->valid) {
        wflow_job_free( wflow_job );
        wflow_job = NULL;
      }
      
      return wflow_job;
    }
  } else {
    config_fprintf_errors( config , stderr );
    exit(1);
  }
}



void wflow_job_free( wflow_job_type * wflow_job ) {
  util_safe_free( wflow_job->module );
  util_safe_free( wflow_job->function );
  util_safe_free( wflow_job->executable );
  int_vector_free( wflow_job->arg_types );
  free( wflow_job->name );
  free( wflow_job );
}


void wflow_job_free__( void * arg) {
  wflow_job_type * wflow_job = wflow_job_safe_cast( arg );
  wflow_job_free( wflow_job );
}
