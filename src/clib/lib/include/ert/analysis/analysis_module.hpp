#ifndef ERT_ANALYSIS_MODULE_H
#define ERT_ANALYSIS_MODULE_H

#include <ert/analysis/ies/ies_config.hpp>

typedef enum {
    ENSEMBLE_SMOOTHER = 1,
    ITERATED_ENSEMBLE_SMOOTHER = 2
} analysis_mode_enum;

typedef struct analysis_module_struct analysis_module_type;

extern "C" analysis_module_type *analysis_module_alloc(analysis_mode_enum mode);
analysis_module_type *analysis_module_alloc_named(analysis_mode_enum mode,
                                                  const char *module_name);

extern "C" void analysis_module_free(analysis_module_type *module);

extern "C" bool analysis_module_set_var(analysis_module_type *module,
                                        const char *var_name,
                                        const char *string_value);
analysis_mode_enum analysis_module_get_mode(const analysis_module_type *module);
extern "C" const char *
analysis_module_get_name(const analysis_module_type *module);

extern "C" bool analysis_module_has_var(const analysis_module_type *module,
                                        const char *var);
extern "C" double analysis_module_get_double(const analysis_module_type *module,
                                             const char *var);
extern "C" int analysis_module_get_int(const analysis_module_type *module,
                                       const char *var);
extern "C" bool analysis_module_get_bool(const analysis_module_type *module,
                                         const char *var);
void *analysis_module_get_ptr(const analysis_module_type *module,
                              const char *var);

ies::Config *
analysis_module_get_module_config(const analysis_module_type *module);

#endif
