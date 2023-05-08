#ifndef ERT_GEN_DATA_CONFIG_H
#define ERT_GEN_DATA_CONFIG_H
#include <stdbool.h>

#include <ert/util/bool_vector.h>
#include <ert/util/stringlist.h>
#include <ert/util/util.h>

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_types.hpp>

typedef struct gen_data_config_struct gen_data_config_type;

extern "C" gen_data_config_type *
gen_data_config_alloc_GEN_DATA_result(const char *key);
extern "C" void gen_data_config_free(gen_data_config_type *);
extern "C" const char *
gen_data_config_get_key(const gen_data_config_type *config);

extern "C" int
gen_data_config_iget_report_step(const gen_data_config_type *config, int index);
void gen_data_config_add_report_step(gen_data_config_type *config,
                                     int report_step);
extern "C" bool
gen_data_config_has_report_step(const gen_data_config_type *config,
                                int report_step);
extern "C" int
gen_data_config_num_report_step(const gen_data_config_type *config);
extern "C" int
gen_data_config_get_data_size__(const gen_data_config_type *config,
                                int report_step);

VOID_FREE_HEADER(gen_data_config)

#endif
