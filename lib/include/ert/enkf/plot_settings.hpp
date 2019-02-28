/*
   Copyright (C) 2011  Equinor ASA, Norway.

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

#include <ert/config/config_parser.hpp>
#include <ert/config/config_settings.hpp>

#ifdef __cplusplus
extern "C" {
#endif

config_settings_type * plot_settings_alloc_load(const char * config_file);
config_settings_type * plot_settings_alloc(const config_content_type * config_content);

void               plot_settings_init(config_settings_type * setting);
void               plot_settings_add_config_items( config_parser_type * config );

#ifdef __cplusplus
}
#endif
#endif
