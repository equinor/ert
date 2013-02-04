/*
   Copyright (C) 2013  Statoil ASA, Norway. 
    
   The file 'analysis_iter_config.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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
#include <string.h>

#include <ert/util/util.h>
#include <ert/util/stringlist.h>

#include <ert/config/config.h>

#include <ert/enkf/enkf_defaults.h>
#include <ert/enkf/config_keys.h>
#include <ert/enkf/analysis_iter_config.h>


struct analysis_iter_config_struct {
  char            * runpath_fmt;
  char            * enspath_fmt;
  stringlist_type * storage;
  int               num_iterations;
};


void analysis_iter_config_set_num_iterations( analysis_iter_config_type * config , int num_iterations) {
  config->num_iterations = num_iterations;    
}

int analysis_iter_config_get_num_iterations( const analysis_iter_config_type * config ) {
  return config->num_iterations;
}



analysis_iter_config_type * analysis_iter_config_alloc() {
   analysis_iter_config_type * config = util_malloc( sizeof * config );    
   config->runpath_fmt    = NULL;
   config->enspath_fmt    = NULL;
   config->storage        = stringlist_alloc_new();
   analysis_iter_config_set_num_iterations( config , DEFAULT_ANALYSIS_NUM_ITERATIONS );
   return config;
}

void analysis_iter_config_free( analysis_iter_config_type * config ) {
  util_safe_free( config->runpath_fmt );
  util_safe_free( config->enspath_fmt );
  stringlist_free( config->storage );
}

static void analysis_iter_config_set_runpath_fmt( analysis_iter_config_type * config , const char * runpath_fmt) {
  config->runpath_fmt = util_realloc_string_copy( config->runpath_fmt , runpath_fmt );
}

static void analysis_iter_config_set_enspath_fmt( analysis_iter_config_type * config , const char * enspath_fmt) {
  config->enspath_fmt = util_realloc_string_copy( config->enspath_fmt , enspath_fmt );
}

const char * analysis_iter_config_iget_runpath_fmt( analysis_iter_config_type * config , int iter) {
  if (config->runpath_fmt != NULL) {
    char * runpath_fmt = util_alloc_sprintf( config->runpath_fmt , iter );
    stringlist_append_owned_ref( config->storage , runpath_fmt );
    return runpath_fmt;
  } else
    return NULL;
}

const char * analysis_iter_config_iget_enspath( analysis_iter_config_type * config , int iter) {
  if (config->enspath_fmt != NULL) {
    char * enspath = util_alloc_sprintf( config->enspath_fmt , iter );
    stringlist_append_owned_ref( config->storage , enspath);
    return enspath;
  } else
    return NULL;
}


void analysis_iter_config_add_config_items( config_type * config ) {
  config_add_key_value( config , ITER_ENSPATH_KEY , false , CONFIG_STRING);
  config_add_key_value( config , ITER_RUNPATH_KEY , false , CONFIG_STRING);
}


void analysis_iter_config_init(analysis_iter_config_type * iter_config , const config_type * config) {
  if (config_item_set( config , ITER_ENSPATH_KEY ))
    analysis_iter_config_set_enspath_fmt( iter_config , config_get_value( config , ITER_ENSPATH_KEY ));
  
  if (config_item_set( config , ITER_RUNPATH_KEY ))
    analysis_iter_config_set_runpath_fmt( iter_config , config_get_value( config , ITER_RUNPATH_KEY ));
}


