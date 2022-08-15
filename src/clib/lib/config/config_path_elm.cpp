/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'config_path_elm.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <filesystem>
#include <string>
#include <vector>

#include <stdlib.h>

#include <ert/util/type_macros.hpp>

#include <ert/config/config_path_elm.hpp>

#include <fmt/format.h>

namespace fs = std::filesystem;

#define CONFIG_PATH_ELM_TYPE_ID 7100063

static UTIL_SAFE_CAST_FUNCTION(config_path_elm, CONFIG_PATH_ELM_TYPE_ID);

config_path_elm_type *config_path_elm_alloc(const fs::path &root_path,
                                            const char *path) {
    auto path_elm = new config_path_elm_type;
    UTIL_TYPE_ID_INIT(path_elm, CONFIG_PATH_ELM_TYPE_ID);
    if (path == NULL) {
        path_elm->path = root_path;
    } else {
        path_elm->path = root_path / path;
    }
    path_elm->path = fs::absolute(path_elm->path);
    return path_elm;
}

void config_path_elm_free(config_path_elm_type *path_elm) { delete path_elm; }

void config_path_elm_free__(void *arg) {
    config_path_elm_type *path_elm = config_path_elm_safe_cast(arg);
    config_path_elm_free(path_elm);
}

const char *config_path_elm_get_abspath(const config_path_elm_type *path_elm) {
    return path_elm->path.c_str();
}

char *config_path_elm_alloc_path(const config_path_elm_type *path_elm,
                                 const char *input_path) {
    if (input_path[0] == '/')
        return strdup(input_path);
    auto path = (path_elm->path / input_path).lexically_normal();
    return strdup(path.c_str());
}
