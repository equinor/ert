/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'config_settings.c' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_CONFIG_SETTINGS_H
#define ERT_CONFIG_SETTINGS_H

#include <stdbool.h>

#include <ert/util/stringlist.hpp>

#include <ert/config/config_parser.hpp>
#include <ert/config/config_content.hpp>
#include <ert/config/config_schema_item.hpp>

typedef struct config_settings_struct config_settings_type;

config_settings_type *config_settings_alloc(const char *root_key);
void config_settings_free(config_settings_type *settings);
bool config_settings_has_key(const config_settings_type *settings,
                             const char *key);
bool config_settings_set_value(const config_settings_type *config_settings,
                               const char *key, const char *value);
void config_settings_init_parser(const config_settings_type *config_settings,
                                 config_parser_type *config, bool required);
void config_settings_init_parser__(const char *root_key,
                                   config_parser_type *config, bool required);
void config_settings_apply(config_settings_type *config_settings,
                           const config_content_type *config);
stringlist_type *
config_settings_alloc_keys(const config_settings_type *config_settings);

bool config_settings_add_setting(config_settings_type *settings,
                                 const char *key, config_item_types value_type,
                                 const char *initial_value);
void config_settings_add_double_setting(config_settings_type *settings,
                                        const char *key, double initial_value);

double
config_settings_get_double_value(const config_settings_type *config_settings,
                                 const char *key);

bool config_settings_set_value(const config_settings_type *config_settings,
                               const char *key, const char *value);
bool config_settings_set_double_value(
    const config_settings_type *config_settings, const char *key, double value);

#endif
