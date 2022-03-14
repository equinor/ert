/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'gen_kw.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_GEN_KW_H
#define ERT_GEN_KW_H

#include <ert/util/double_vector.h>
#include <ert/res_util/subst_list.hpp>
#include <ert/tooling.hpp>

#include <ert/enkf/gen_kw_config.hpp>
#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_serialize.hpp>
#include <ert/enkf/gen_kw_common.hpp>

extern "C" void gen_kw_ecl_write(const gen_kw_type *gen_kw,
                                 const char *run_path, const char *base_file,
                                 value_export_type *export_value);
extern "C" PY_USED void gen_kw_write_export_file(const gen_kw_type *gen_kw,
                                                 const char *filename);

extern "C" void gen_kw_free(gen_kw_type *);
extern "C" int gen_kw_data_size(const gen_kw_type *);
extern "C" double gen_kw_data_iget(const gen_kw_type *, int, bool);
extern "C" void gen_kw_data_iset(gen_kw_type *, int, double);
extern "C" PY_USED void
gen_kw_data_set_vector(gen_kw_type *gen_kw, const double_vector_type *values);
extern "C" double gen_kw_data_get(gen_kw_type *, const char *, bool);
extern "C" void gen_kw_data_set(gen_kw_type *, const char *, double);
extern "C" PY_USED bool gen_kw_data_has_key(gen_kw_type *, const char *);
extern "C" const char *gen_kw_get_name(const gen_kw_type *, int);
void gen_kw_filter_file(const gen_kw_type *, const char *);
extern "C" void gen_kw_ecl_write_template(const gen_kw_type *gen_kw,
                                          const char *file_name);

VOID_ECL_WRITE_HEADER(gen_kw)
VOID_COPY_HEADER(gen_kw);
VOID_INITIALIZE_HEADER(gen_kw);
VOID_FREE_HEADER(gen_kw);
VOID_ALLOC_HEADER(gen_kw);
VOID_ECL_WRITE_HEADER(gen_kw);
VOID_USER_GET_HEADER(gen_kw);
VOID_WRITE_TO_BUFFER_HEADER(gen_kw);
VOID_READ_FROM_BUFFER_HEADER(gen_kw);
VOID_FLOAD_HEADER(gen_kw);
VOID_CLEAR_HEADER(gen_kw);
VOID_SERIALIZE_HEADER(gen_kw)
VOID_DESERIALIZE_HEADER(gen_kw)
#endif
