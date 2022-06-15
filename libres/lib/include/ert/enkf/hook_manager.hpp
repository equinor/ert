/*
   Copyright (C) 2012  Equinor ASA, Norway.

   The file 'hook_manager.h' is part of ERT - Ensemble based Reservoir Tool.

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
#ifndef ERT_HOOK_MANAGER_H
#define ERT_HOOK_MANAGER_H

#include <ert/config/config_content.hpp>
#include <ert/config/config_parser.hpp>

#include <ert/enkf/ert_run_context.hpp>
#include <ert/enkf/ert_workflow_list.hpp>
#include <ert/enkf/hook_workflow.hpp>
#include <vector>

#define RUNPATH_LIST_FILE ".ert_runpath_list"

typedef struct hook_manager_struct hook_manager_type;

hook_manager_type *
hook_manager_alloc_default(ert_workflow_list_type *workflow_list);
extern "C" hook_manager_type *hook_manager_alloc(ert_workflow_list_type *,
                                                 const config_content_type *);

extern "C" PY_USED hook_manager_type *hook_manager_alloc_full(
    ert_workflow_list_type *workflow_list, const char *runpath_list_file,
    const char **hook_workflow_names, const char **hook_workflow_run_modes,
    int hook_workflow_count);

extern "C" void hook_manager_free(hook_manager_type *hook_manager);

void hook_manager_init(hook_manager_type *hook_manager,
                       const config_content_type *config);
void hook_manager_add_config_items(config_parser_type *config);

extern "C" const char *
hook_manager_get_runpath_list_file(const hook_manager_type *hook_manager);
void hook_manager_run_workflows(const hook_manager_type *hook_manager,
                                hook_run_mode_enum run_mode, void *self);

extern "C" PY_USED const hook_workflow_type *
hook_manager_iget_hook_workflow(const hook_manager_type *hook_manager,
                                int index);
extern "C" int hook_manager_get_size(const hook_manager_type *hook_manager);

/** Runpath contains the data for one line in the runpath list file */
class Runpath {
public:
    /**
     * @param iens The ensamble index of the run
     * @param iter The iteration of the run
     * @param runpath The absolute path to the run,
     *     ie. <...>/poly_example/poly_out/realization-0/iter-0
     * @param jobname The name of the job
     */
    explicit Runpath(int iens, int iter, std::string runpath,
                     std::string jobname)
        : iens(iens), iter(iter), runpath(runpath), jobname(jobname) {}
    explicit Runpath(int iens, int iter, const char *runpath_,
                     const char *jobname_)
        : iens(iens), iter(iter) {
        if (jobname_ == NULL) {
            jobname = "";
        } else {
            jobname = std::string(jobname_);
        }

        if (runpath_ == NULL) {
            runpath = "";
        } else {
            runpath = std::string(runpath_);
        }
    }
    /** The index of the run in the ensemble */
    int iens;
    /** The index of the iteration */
    int iter;
    /** Absolute path of the runs directory */
    std::string runpath;
    std::string jobname;

    bool operator<(const Runpath &other) const {
        if (iter < other.iter)
            return true;
        if (iter > other.iter)
            return false;
        return iens < other.iens;
    }
};

void hook_manager_write_runpath_file(const hook_manager_type *hook_manager,
                                     std::vector<Runpath> runpath_list);
#endif
