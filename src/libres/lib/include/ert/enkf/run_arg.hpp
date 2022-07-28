/*
   Copyright (C) 2014  Equinor ASA, Norway.

   The file 'run_arg.c' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_RUN_ARG_H
#define ERT_RUN_ARG_H

#include <ert/res_util/path_fmt.hpp>
#include <ert/util/type_macros.h>

#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/run_arg_type.hpp>

UTIL_SAFE_CAST_HEADER(run_arg);
UTIL_IS_INSTANCE_HEADER(run_arg);

extern "C" run_arg_type *run_arg_alloc(const char *run_id, enkf_fs_type *sim_fs, int iens,
                                   run_mode_type run_mode,
                                   int iter, const char *runpath,
                                   const char *job_name);

extern "C" run_arg_type *
run_arg_alloc_ENSEMBLE_EXPERIMENT(const char *run_id, enkf_fs_type *sim_fs,
                                  int iens, int iter, const char *runpath,
                                  const char *job_name);

run_arg_type *run_arg_alloc_INIT_ONLY(const char *run_id, enkf_fs_type *sim_fs,
                                      int iens, int iter, const char *runpath);

run_arg_type *
run_arg_alloc_SMOOTHER_RUN(const char *run_id, enkf_fs_type *sim_fs, int iens, int iter,
                           const char *runpath, const char *job_name);

extern "C" int run_arg_get_iens(const run_arg_type *run_arg);
extern "C" int run_arg_get_iter(const run_arg_type *run_arg);
extern "C" void run_arg_set_queue_index(run_arg_type *run_arg, int queue_index);

extern "C" void run_arg_free(run_arg_type *run_arg);
void run_arg_free__(void *arg);
extern "C" const char *run_arg_get_job_name(const run_arg_type *run_arg);
extern "C" const char *run_arg_get_runpath(const run_arg_type *run_arg);
extern "C" const char *run_arg_get_run_id(const run_arg_type *run_arg);
extern "C" run_status_type run_arg_get_run_status(const run_arg_type *run_arg);

extern "C" int run_arg_get_queue_index_safe(const run_arg_type *run_arg);
int run_arg_get_queue_index(const run_arg_type *run_arg);
extern "C" bool run_arg_is_submitted(const run_arg_type *run_arg);

bool run_arg_can_retry(const run_arg_type *run_arg);

run_status_type run_arg_get_run_status(const run_arg_type *run_arg);
void run_arg_set_run_status(run_arg_type *run_arg, run_status_type run_status);

enkf_fs_type *run_arg_get_sim_fs(const run_arg_type *run_arg);
#endif
