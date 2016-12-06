/*
   Copyright (C) 2011  Statoil ASA, Norway.

   The file 'plot_settings.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_PLOT_SETTINGS_H
#define ERT_PLOT_SETTINGS_H

#include <ert/util/stringlist.h>

#include <ert/config/config_parser.h>
#include <ert/config/config_content.h>
#include <ert/config/config_schema_item.h>


typedef struct plot_settings_struct plot_settings_type;

void               plot_settings_set_path(plot_settings_type * plot_settings , const char * plot_path);
const char  *      plot_settings_get_path(const plot_settings_type * plot_settings );
void               plot_settings_free( plot_settings_type * plot_settings);
plot_settings_type * plot_settings_alloc();
void               plot_settings_init(plot_settings_type * plot_settings , const config_content_type * config );
void               plot_settings_add_config_items( config_parser_type * config );
stringlist_type  * plot_setting_alloc_keys( const plot_settings_type * plot_settings );
bool               plot_settings_has_key( const plot_settings_type * plot_settings , const char * key);
config_item_types  plot_settings_get_value_type( const plot_settings_type * plot_settings , const char * key);
bool               plot_settings_set_value( const plot_settings_type * plot_settings , const char * key, const char * value);
const char       * plot_settings_get_value( const plot_settings_type * plot_settings , const char * key);

#endif
