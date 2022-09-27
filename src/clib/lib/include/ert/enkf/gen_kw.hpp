#ifndef ERT_GEN_KW_H
#define ERT_GEN_KW_H

#include <ert/res_util/subst_list.hpp>
#include <ert/tooling.hpp>
#include <ert/util/double_vector.h>

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_serialize.hpp>
#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/gen_kw_common.hpp>
#include <ert/enkf/gen_kw_config.hpp>

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

UTIL_SAFE_CAST_HEADER(gen_kw);
UTIL_SAFE_CAST_HEADER_CONST(gen_kw);
VOID_ECL_WRITE_HEADER(gen_kw)
VOID_COPY_HEADER(gen_kw);
VOID_FREE_HEADER(gen_kw);
VOID_ALLOC_HEADER(gen_kw);
VOID_ECL_WRITE_HEADER(gen_kw);
VOID_USER_GET_HEADER(gen_kw);
VOID_WRITE_TO_BUFFER_HEADER(gen_kw);
VOID_READ_FROM_BUFFER_HEADER(gen_kw);
VOID_SERIALIZE_HEADER(gen_kw)
VOID_DESERIALIZE_HEADER(gen_kw)
#endif
