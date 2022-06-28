/*
   Copyright (C) 2014  Equinor ASA, Norway.

   The file 'ert_run_context.c' is part of ERT - Ensemble based Reservoir Tool.

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
#include <time.h>
#include <unistd.h>

#include <ert/util/stringlist.h>
#include <ert/util/type_macros.h>
#include <ert/util/type_vector_functions.h>
#include <ert/util/vector.h>

#include <ert/res_util/path_fmt.hpp>

#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/ert_run_context.hpp>
#include <ert/enkf/run_arg.hpp>

#include <ert/python.hpp>

#define ERT_RUN_CONTEXT_TYPE_ID 55534132

struct ert_run_context_struct {
    UTIL_TYPE_ID_DECLARATION;
    vector_type *run_args;
    run_mode_type run_mode;
    init_mode_type init_mode;
    int iter;
    int step1;
    int step2;
    int load_start;
    std::vector<bool> active;
    enkf_fs_type *sim_fs;
    enkf_fs_type *update_target_fs;
    char *run_id;
};

char *ert_run_context_alloc_run_id() {
    int year, month, day, hour, min, sec;
    time_t now = time(NULL);
    unsigned int random = util_dev_urandom_seed();
    util_set_datetime_values_utc(now, &sec, &min, &hour, &day, &month, &year);
    return util_alloc_sprintf("%d:%d:%4d-%0d-%02d-%02d-%02d-%02d:%ud", getpid(),
                              getuid(), year, month, day, hour, min, sec,
                              random);
}

extern "C" ert_run_context_type *
ert_run_context_alloc_empty(run_mode_type run_mode, init_mode_type init_mode,
                            int iter) {
    auto context = new ert_run_context_type;
    UTIL_TYPE_ID_INIT(context, ERT_RUN_CONTEXT_TYPE_ID);

    context->run_args = vector_alloc_new();
    context->run_mode = run_mode;
    context->init_mode = init_mode;
    context->iter = iter;

    context->sim_fs = nullptr;
    context->update_target_fs = nullptr;

    context->step1 = 0;
    context->step2 = 0;
    context->run_id = ert_run_context_alloc_run_id();
    return context;
}

void ert_run_context_set_sim_fs(ert_run_context_type *context,
                                enkf_fs_type *sim_fs) {
    if (sim_fs) {
        context->sim_fs = sim_fs;
        enkf_fs_increase_run_count(sim_fs);
        enkf_fs_incref(sim_fs);
    } else
        context->sim_fs = NULL;
}

void ert_run_context_add_ENSEMBLE_EXPERIMENT_args(
    ert_run_context_type *context, std::vector<std::string> runpaths,
    std::vector<std::string> jobnames) {
    for (int iens = 0; iens < context->active.size(); iens++) {
        if (context->active[iens]) {
            run_arg_type *arg = run_arg_alloc_ENSEMBLE_EXPERIMENT(
                context->run_id, context->sim_fs, iens, context->iter,
                runpaths[iens].c_str(), jobnames[iens].c_str());
            vector_append_owned_ref(context->run_args, arg, run_arg_free__);
        } else
            vector_append_ref(context->run_args, NULL);
    }
}

ert_run_context_type *ert_run_context_alloc_ENSEMBLE_EXPERIMENT(
    enkf_fs_type *sim_fs, std::vector<bool> active,
    std::vector<std::string> runpaths, std::vector<std::string> jobnames,
    int iter) {

    ert_run_context_type *context = ert_run_context_alloc_empty(
        ENSEMBLE_EXPERIMENT, INIT_CONDITIONAL, iter);
    context->active = active;
    ert_run_context_set_sim_fs(context, sim_fs);
    ert_run_context_add_ENSEMBLE_EXPERIMENT_args(context, runpaths, jobnames);
    return context;
}

static void
ert_run_context_add_INIT_ONLY_args(ert_run_context_type *context,
                                   std::vector<std::string> runpaths) {
    for (int iens = 0; iens < context->active.size(); iens++) {
        if (context->active[iens]) {
            run_arg_type *arg =
                run_arg_alloc_INIT_ONLY(context->run_id, context->sim_fs, iens,
                                        context->iter, runpaths[iens].c_str());
            vector_append_owned_ref(context->run_args, arg, run_arg_free__);
        } else
            vector_append_ref(context->run_args, NULL);
    }
}

ert_run_context_type *
ert_run_context_alloc_INIT_ONLY(enkf_fs_type *sim_fs, init_mode_type init_mode,
                                std::vector<bool> active,
                                std::vector<std::string> runpaths, int iter) {
    ert_run_context_type *context =
        ert_run_context_alloc_empty(INIT_ONLY, init_mode, iter);
    context->active = active;
    ert_run_context_set_sim_fs(context, sim_fs);

    ert_run_context_add_INIT_ONLY_args(context, runpaths);
    return context;
}

static void
ert_run_context_set_update_target_fs(ert_run_context_type *context,
                                     enkf_fs_type *update_target_fs) {
    if (update_target_fs) {
        context->update_target_fs = update_target_fs;
        enkf_fs_increase_run_count(update_target_fs);
        enkf_fs_incref(update_target_fs);
    } else
        context->update_target_fs = NULL;
}

static void
ert_run_context_add_SMOOTHER_RUN_args(ert_run_context_type *context,
                                      std::vector<std::string> runpaths,
                                      std::vector<std::string> jobnames) {

    for (int iens = 0; iens < context->active.size(); iens++) {
        if (context->active[iens]) {
            run_arg_type *arg = run_arg_alloc_SMOOTHER_RUN(
                context->run_id, context->sim_fs, context->update_target_fs,
                iens, context->iter, runpaths[iens].c_str(),
                jobnames[iens].c_str());
            vector_append_owned_ref(context->run_args, arg, run_arg_free__);
        } else
            vector_append_ref(context->run_args, NULL);
    }
}

ert_run_context_type *ert_run_context_alloc_SMOOTHER_RUN(
    enkf_fs_type *sim_fs, enkf_fs_type *target_fs, std::vector<bool> active,
    std::vector<std::string> runpaths, std::vector<std::string> jobnames,
    int iter) {

    ert_run_context_type *context =
        ert_run_context_alloc_empty(SMOOTHER_RUN, INIT_CONDITIONAL, iter);
    context->active = active;
    ert_run_context_set_sim_fs(context, sim_fs);
    ert_run_context_set_update_target_fs(context, target_fs);
    ert_run_context_add_SMOOTHER_RUN_args(context, runpaths, jobnames);
    return context;
}

UTIL_IS_INSTANCE_FUNCTION(ert_run_context, ERT_RUN_CONTEXT_TYPE_ID);

const char *ert_run_context_get_id(const ert_run_context_type *context) {
    return context->run_id;
}

void ert_run_context_free(ert_run_context_type *context) {
    if (context->sim_fs) {
        enkf_fs_decrease_run_count(context->sim_fs);
        enkf_fs_decref(context->sim_fs);
    }

    if (context->update_target_fs) {
        enkf_fs_decrease_run_count(context->update_target_fs);
        enkf_fs_decref(context->update_target_fs);
    }

    vector_free(context->run_args);
    free(context->run_id);
    delete context;
}

int ert_run_context_get_size(const ert_run_context_type *context) {
    return vector_get_size(context->run_args);
}

int ert_run_context_get_iter(const ert_run_context_type *context) {
    return context->iter;
}

init_mode_type
ert_run_context_get_init_mode(const ert_run_context_type *context) {
    return context->init_mode;
}

int ert_run_context_get_step1(const ert_run_context_type *context) {
    return context->step1;
}

run_arg_type *ert_run_context_iget_arg(const ert_run_context_type *context,
                                       int index) {
    return (run_arg_type *)vector_iget(context->run_args, index);
}

enkf_fs_type *
ert_run_context_get_sim_fs(const ert_run_context_type *run_context) {
    if (run_context->sim_fs)
        return run_context->sim_fs;
    else {
        util_abort("%s: internal error - tried to access run_context->sim_fs "
                   "when sim_fs == NULL\n",
                   __func__);
        return NULL;
    }
}

enkf_fs_type *
ert_run_context_get_update_target_fs(const ert_run_context_type *run_context) {
    if (run_context->update_target_fs)
        return run_context->update_target_fs;
    else {
        util_abort(
            "%s: internal error - tried to access "
            "run_context->update_target_fs when update_target_fs == NULL\n",
            __func__);
        return NULL;
    }
}

void ert_run_context_deactivate_realization(ert_run_context_type *context,
                                            int iens) {
    context->active[iens] = false;
}

bool ert_run_context_iactive(const ert_run_context_type *context, int iens) {
    return context->active[iens];
}

RES_LIB_SUBMODULE("ert_run_context", m) {
    using namespace py::literals;
    m.def("set_sim_fs", [](py::object self, py::object sim_fs) {
        ert_run_context_set_sim_fs(ert::from_cwrap<ert_run_context_type>(self),
                                   ert::from_cwrap<enkf_fs_type>(sim_fs));
    });
    m.def("set_target_fs", [](py::object self, py::object target_fs) {
        ert_run_context_set_update_target_fs(
            ert::from_cwrap<ert_run_context_type>(self),
            ert::from_cwrap<enkf_fs_type>(target_fs));
    });
    m.def("set_active", [](py::object self, std::vector<bool> active) {
        ert::from_cwrap<ert_run_context_type>(self)->active = active;
    });
    m.def("add_ensemble_experiment_args",
          [](py::object self, std::vector<std::string> runpaths,
             std::vector<std::string> jobnames) {
              ert_run_context_add_ENSEMBLE_EXPERIMENT_args(
                  ert::from_cwrap<ert_run_context_type>(self), runpaths,
                  jobnames);
          });
    m.def("add_init_only_args",
          [](py::object self, std::vector<std::string> runpaths) {
              ert_run_context_add_INIT_ONLY_args(
                  ert::from_cwrap<ert_run_context_type>(self), runpaths);
          });
    m.def("add_smoother_run_args", [](py::object self,
                                      std::vector<std::string> runpaths,
                                      std::vector<std::string> jobnames) {
        ert_run_context_add_SMOOTHER_RUN_args(
            ert::from_cwrap<ert_run_context_type>(self), runpaths, jobnames);
    });
}
