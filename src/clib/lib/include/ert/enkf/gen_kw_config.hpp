#ifndef ERT_GEN_KW_CONFIG_H
#define ERT_GEN_KW_CONFIG_H

#include <stdbool.h>
#include <stdio.h>

#include <ert/util/stringlist.h>
#include <ert/util/util.h>

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_util.hpp>

typedef struct gen_kw_config_struct gen_kw_config_type;

extern "C" const char *
gen_kw_config_get_tag_fmt(const gen_kw_config_type *config);
bool gen_kw_config_is_valid(const gen_kw_config_type *config);
void gen_kw_config_fprintf_config(const gen_kw_config_type *config,
                                  const char *outfile, FILE *stream);
extern "C" void gen_kw_config_set_parameter_file(gen_kw_config_type *config,
                                                 const char *parameter_file);
extern "C" PY_USED const char *
gen_kw_config_get_parameter_file(const gen_kw_config_type *config);
extern "C" const char *gen_kw_config_get_key(const gen_kw_config_type *config);
extern "C" const char *
gen_kw_config_get_template_file(const gen_kw_config_type *);
extern "C" void gen_kw_config_free(gen_kw_config_type *);
extern "C" PY_USED double gen_kw_config_transform(const gen_kw_config_type *,
                                                  int index, double x);
extern "C" bool
gen_kw_config_should_use_log_scale(const gen_kw_config_type *config, int index);
extern "C" int gen_kw_config_get_data_size(const gen_kw_config_type *);
extern "C" const char *gen_kw_config_iget_name(const gen_kw_config_type *, int);
const char *gen_kw_config_get_tagged_name(const gen_kw_config_type *, int);
extern "C" stringlist_type *
gen_kw_config_alloc_name_list(const gen_kw_config_type *config);
int gen_kw_config_get_index(const gen_kw_config_type *, const char *);
extern "C" void gen_kw_config_set_template_file(gen_kw_config_type *config,
                                                const char *template_file);
extern "C" gen_kw_config_type *gen_kw_config_alloc_empty(const char *key,
                                                         const char *tag_fmt);
void gen_kw_config_update(gen_kw_config_type *config, const char *template_file,
                          const char *parameter_file);
void gen_kw_config_update_tag_format(gen_kw_config_type *config,
                                     const char *tag_format);
extern "C" PY_USED const char *
gen_kw_config_iget_function_type(const gen_kw_config_type *config, int index);
extern "C" double_vector_type *
gen_kw_config_iget_function_parameter_values(const gen_kw_config_type *config,
                                             int index);
extern "C" stringlist_type *
gen_kw_config_iget_function_parameter_names(const gen_kw_config_type *config,
                                            int index);

VOID_FREE_HEADER(gen_kw_config);
VOID_GET_DATA_SIZE_HEADER(gen_kw);
#endif
