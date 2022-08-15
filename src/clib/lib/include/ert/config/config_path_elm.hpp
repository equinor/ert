/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'config_path_elm.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_CONFIG_PATH_ELM_H
#define ERT_CONFIG_PATH_ELM_H

#include <filesystem>

struct config_path_elm_type {
    /** UTIL_TYPE_ID_DECLARATION */
    int __type_id;
    std::filesystem::path path;
};

extern "C" void config_path_elm_free(config_path_elm_type *path_elm);
void config_path_elm_free__(void *arg);
config_path_elm_type *
config_path_elm_alloc(const std::filesystem::path &root_path, const char *path);
extern "C" const char *
config_path_elm_get_abspath(const config_path_elm_type *path_elm);
char *config_path_elm_alloc_path(const config_path_elm_type *path_elm,
                                 const char *input_path);
#define config_path_elm_alloc_abspath(path_elm, input_path)                    \
    config_path_elm_alloc_path(path_elm, input_path)

#endif
