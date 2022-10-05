#ifndef ERT_ANALYSIS_CONFIG_H
#define ERT_ANALYSIS_CONFIG_H

#include <string>
#include <vector>

#include <stdbool.h>

#include <ert/util/stringlist.h>

#include <ert/config/config_content.hpp>
#include <ert/config/config_parser.hpp>

#include <ert/analysis/analysis_module.hpp>

#include <ert/enkf/enkf_types.hpp>

typedef struct analysis_config_struct analysis_config_type;

extern "C" analysis_module_type *
analysis_config_get_module(const analysis_config_type *config,
                           const char *module_name);
extern "C" bool analysis_config_has_module(const analysis_config_type *config,
                                           const char *module_name);
std::vector<std::string>
analysis_config_module_names(const analysis_config_type *config);

extern "C" PY_USED analysis_config_type *analysis_config_alloc();
extern "C" void analysis_config_free(analysis_config_type *);
extern "C" bool analysis_config_select_module(analysis_config_type *config,
                                              const char *module_name);
analysis_module_type *
analysis_config_get_active_module(const analysis_config_type *config);

extern "C" PY_USED const char *
analysis_config_get_active_module_name(const analysis_config_type *config);

extern "C" PY_USED void
analysis_config_add_module_copy(analysis_config_type *config,
                                const char *src_name, const char *target_name);
void analysis_config_load_internal_modules(int ens_size,
                                           analysis_config_type *config);

#endif
