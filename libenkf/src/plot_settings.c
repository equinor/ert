/*
   Copyright (C) 2011  Statoil ASA, Norway.

   The file 'plot_config.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/util/util.h>
#include <ert/util/stringlist.h>
#include <ert/util/hash.h>

#include <ert/config/config_parser.h>
#include <ert/config/config_schema_item.h>
#include <ert/config/config_content.h>

#include <ert/enkf/plot_settings.h>
#include <ert/enkf/enkf_defaults.h>
#include <ert/enkf/config_keys.h>

#define TRUE_STRING              "True"
#define FALSE_STRING             "False"

#define PATH_KEY                 "PATH"
#define SHOW_REFCASE_KEY         "SHOW_REFCASE"
#define SHOW_HISTORY_KEY         "SHOW_HISTORY"

#define DEFAULT_PLOT_PATH        "plots"
#define DEFAULT_SHOW_REFCASE     FALSE_STRING
#define DEFAULT_SHOW_HISTORY     FALSE_STRING


/**
    Struct holding basic information used when plotting.
*/

struct plot_settings_struct {
  hash_type * settings;
  hash_type * value_types;
};

void plot_settings_set_path(plot_settings_type * plot_settings , const char * plot_path) {
  plot_settings_set_value(plot_settings,PATH_KEY, plot_path);
}

const char *  plot_settings_get_path(const plot_settings_type * plot_settings ) {
  return plot_settings_get_value(plot_settings, PATH_KEY);
}

void plot_settings_free( plot_settings_type * plot_settings) {
  hash_free( plot_settings->settings );
  hash_free( plot_settings->value_types );
  free(plot_settings);
}


static void plot_settings_register_key( plot_settings_type * plot_settings , const char * key , const char * value, config_item_types value_type)
{
  hash_insert_ref( plot_settings->settings , key , value );
  hash_insert_int( plot_settings->value_types , key , value_type );
}


/**
   The plot_settings object is instantiated with the default values from enkf_defaults.h
*/
plot_settings_type * plot_settings_alloc() {
  plot_settings_type * info        = util_malloc( sizeof * info );
  info->settings = hash_alloc();
  info->value_types = hash_alloc();

  plot_settings_register_key( info , PATH_KEY , DEFAULT_PLOT_PATH , CONFIG_STRING);
  plot_settings_register_key( info , SHOW_REFCASE_KEY , DEFAULT_SHOW_REFCASE , CONFIG_BOOL);
  plot_settings_register_key( info , SHOW_HISTORY_KEY , DEFAULT_SHOW_HISTORY , CONFIG_BOOL);

  return info;
}


stringlist_type * plot_settings_alloc_keys( const plot_settings_type * plot_settings ) {
  return hash_alloc_stringlist(plot_settings->settings);
}


bool plot_settings_has_key( const plot_settings_type * plot_settings , const char * key) {
  return hash_has_key( plot_settings->settings , key );
}


const char * plot_settings_get_value( const plot_settings_type * plot_settings , const char * key) {
  return hash_get( plot_settings->settings , key );
}


config_item_types plot_settings_get_value_type( const plot_settings_type * plot_settings , const char * key) {
  return hash_get_int( plot_settings->value_types , key );
}


bool plot_settings_set_value( const plot_settings_type * plot_settings , const char * key, const char * value) {
  if (hash_has_key( plot_settings->value_types , key )) {
    config_item_types value_type = hash_get_int( plot_settings->value_types, key );
    if (config_schema_item_valid_string(value_type , value)) {
      hash_insert_hash_owned_ref( plot_settings->settings , key , util_alloc_string_copy( value ), free);
      return true;
    } else
      return false;
  }

  return false;
}



void plot_settings_init(plot_settings_type * plot_settings , const config_content_type * config ) {
  if (config_content_has_item( config , PLOT_PATH_KEY))
    plot_settings_set_path( plot_settings , config_content_get_value( config , PLOT_PATH_KEY ));

  for (int i = 0; i < config_content_get_occurences(config, PLOT_SETTING_KEY); i++) {
      const stringlist_type * tokens = config_content_iget_stringlist_ref(config, PLOT_SETTING_KEY, i);
      const char * setting = stringlist_iget(tokens, 0);
      const char * value = stringlist_iget(tokens, 1);

      bool set_ok = plot_settings_set_value( plot_settings , setting , value );
      if (!set_ok)
        fprintf(stderr," ** Warning: failed to apply PLOT_SETTING %s=%s \n",setting,value);
  }
}



void plot_settings_add_config_items( config_parser_type * config ) {
  config_schema_item_type * item = config_add_schema_item(config, PLOT_SETTING_KEY, false);
  config_schema_item_set_argc_minmax(item, 2, 2);

  config_add_key_value(config , PLOT_PATH_KEY         , false , CONFIG_STRING);
  {
    char * msg = util_alloc_sprintf( "The keyword %s has been deprecated - use %s %s <PATH>", PLOT_PATH_KEY , PLOT_SETTING_KEY , PATH_KEY );
    config_parser_deprecate( config , PLOT_PATH_KEY , msg);
    free( msg );
  }
}


