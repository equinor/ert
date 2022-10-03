/*
   Copyright (C) 2017  Equinor ASA, Norway.

   The file 'environment_varlist.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <ert/job_queue/environment_varlist.hpp>

#include <ert/res_util/res_env.hpp>

#include <map>

#include <ert/python.hpp>

#define ENV_VAR_KEY_STRING "global_environment"
#define UPDATE_PATH_KEY_STRING "global_update_path"

struct env_varlist_struct {
    std::map<std::string, std::string> varlist;
    std::map<std::string, std::string> updatelist;
};

env_varlist_type *env_varlist_alloc() { return new env_varlist_struct; }

void env_varlist_update_path(env_varlist_type *list, const char *path_var,
                             const char *new_path) {
    list->updatelist[path_var] =
        res_env_update_path_var(path_var, new_path, false);
}

void env_varlist_setenv(env_varlist_type *list, const char *key,
                        const char *value) {
    list->varlist[key] = res_env_interp_setenv(key, value);
}

static void print_map_as_json(const std::map<std::string, std::string> &map,
                              FILE *stream) {
    bool first = true;
    fprintf(stream, "{");
    for (const auto &[key, value] : map) {
        if (!first) {
            fprintf(stream, ", ");
        }
        first = false;
        fprintf(stream, R"("%s" : "%s")", key.c_str(), value.c_str());
    }
    fprintf(stream, "}");
}

void env_varlist_json_fprintf(const env_varlist_type *list, FILE *stream) {
    fprintf(stream, "\"%s\" : ", ENV_VAR_KEY_STRING);
    print_map_as_json(list->varlist, stream);
    fprintf(stream, ",\n");
    fprintf(stream, "\"%s\" : ", UPDATE_PATH_KEY_STRING);
    print_map_as_json(list->updatelist, stream);
}

void env_varlist_free(env_varlist_type *list) { delete list; }

ERT_CLIB_SUBMODULE("env_varlist", m) {
    using namespace py::literals;
    m.def(
        "_get_varlist",
        [](Cwrap<env_varlist_type> self) { return self->varlist; },
        py::arg("self"));
    m.def(
        "_get_updatelist",
        [](Cwrap<env_varlist_type> self) { return self->updatelist; },
        py::arg("self"));
}
