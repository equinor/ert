/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'ecl_config.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_ECL_CONFIG_H
#define ERT_ECL_CONFIG_H
#include <time.h>

#include <ert/res_util/ui_return.hpp>

#include <ert/config/config_content.hpp>
#include <ert/config/config_parser.hpp>

#include <ert/ecl/ecl_grid.hpp>
#include <ert/ecl/ecl_io_config.hpp>
#include <ert/ecl/ecl_sum.hpp>

#include <ert/res_util/path_fmt.hpp>

#include <ert/enkf/ecl_refcase_list.hpp>

typedef struct ecl_config_struct ecl_config_type;

extern "C" const char *ecl_config_get_data_file(const ecl_config_type *);
extern "C" void ecl_config_set_data_file(ecl_config_type *ecl_config,
                                         const char *data_file);
extern "C" ui_return_type *
ecl_config_validate_data_file(const ecl_config_type *ecl_config,
                              const char *data_file);

ui_return_type *ecl_config_validate_eclbase(const ecl_config_type *ecl_config,
                                            const char *eclbase_fmt);

extern "C" void ecl_config_set_grid(ecl_config_type *ecl_config,
                                    const char *grid_file);
extern "C" const char *
ecl_config_get_gridfile(const ecl_config_type *ecl_config);
extern "C" ecl_grid_type *ecl_config_get_grid(const ecl_config_type *);
extern "C" ui_return_type *
ecl_config_validate_grid(const ecl_config_type *ecl_config,
                         const char *grid_file);

extern "C" bool ecl_config_load_refcase(ecl_config_type *ecl_config,
                                        const char *refcase);
extern "C" ui_return_type *
ecl_config_validate_refcase(const ecl_config_type *ecl_config,
                            const char *refcase);
extern "C" const ecl_sum_type *
ecl_config_get_refcase(const ecl_config_type *ecl_config);
extern "C" bool ecl_config_has_refcase(const ecl_config_type *ecl_config);
ecl_refcase_list_type *
ecl_config_get_refcase_list(const ecl_config_type *ecl_config);

extern "C" bool ecl_config_active(const ecl_config_type *config);
extern "C" time_t ecl_config_get_end_date(const ecl_config_type *ecl_config);

const char *
ecl_config_get_schedule_prediction_file(const ecl_config_type *ecl_config);
void ecl_config_set_schedule_prediction_file(
    ecl_config_type *ecl_config, const char *schedule_prediction_file);

extern "C" int ecl_config_get_num_cpu(const ecl_config_type *ecl_config);
void ecl_config_init(ecl_config_type *ecl_config,
                     const config_content_type *config);
extern "C" void ecl_config_free(ecl_config_type *);

bool ecl_config_get_formatted(const ecl_config_type *);
bool ecl_config_get_unified_restart(const ecl_config_type *);
extern "C" int ecl_config_get_last_history_restart(const ecl_config_type *);
bool ecl_config_can_restart(const ecl_config_type *ecl_config);
void ecl_config_assert_restart(const ecl_config_type *ecl_config);
extern "C" const char *
ecl_config_get_refcase_name(const ecl_config_type *ecl_config);
extern "C" ecl_config_type *
ecl_config_alloc(const config_content_type *config_content);
extern "C" PY_USED ecl_config_type *
ecl_config_alloc_full(bool have_eclbase, char *data_file, ecl_grid_type *grid,
                      char *refcase_default, stringlist_type *ref_case_list,
                      time_t end_date, char *sched_prediction_file);
void ecl_config_add_config_items(config_parser_type *config);
extern "C" const char *
ecl_config_get_depth_unit(const ecl_config_type *ecl_config);
extern "C" const char *
ecl_config_get_pressure_unit(const ecl_config_type *ecl_config);
bool ecl_config_have_eclbase(const ecl_config_type *ecl_config);

#endif
