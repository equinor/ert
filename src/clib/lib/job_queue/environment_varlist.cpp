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
