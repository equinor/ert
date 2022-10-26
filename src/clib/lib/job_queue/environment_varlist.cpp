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
