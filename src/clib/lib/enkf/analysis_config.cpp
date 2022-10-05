#include <string>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string>

#include <ert/util/util.h>

#include <ert/config/config_parser.hpp>

#include <ert/analysis/analysis_module.hpp>

#include <ert/enkf/analysis_config.hpp>
#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/model_config.hpp>
#include <ert/except.hpp>

using namespace std::string_literals;

struct analysis_config_struct {
    std::unordered_map<std::string, analysis_module_type *> analysis_modules;
    analysis_module_type *analysis_module;
};

/*

Interacting with modules
------------------------

It is possible to create a copy of an analysis module under a different
name, this can be convenient when trying out the same algorithm with
different parameter settings. I.e. based on the built in module STD_ENKF we
can create two copies with high and low truncation respectively:

   ANALYSIS_COPY  STD_ENKF  ENKF_HIGH_TRUNCATION
   ANALYSIS_COPY  STD_ENKF  ENKF_LOW_TRUNCATION

The copy operation does not differentiate between external and internal
modules. When a module has been loaded you can set internal parameters for
the module with the config command:

   ANALYSIS_SET_VAR  ModuleName  VariableName   Value

The module will be called with a function for setting variables which gets
the VariableName and value parameters as input; if the module recognizes
VariableName and Value is of the right type the module should set the
internal variable accordingly. If the module does not recognize the
variable name a warning will be printed on stderr, but no further action.

The actual analysis module to use is selected with the statement:

ANALYSIS_SELECT  ModuleName

[1] The libfile argument should include the '.so' extension, and can
    optionally contain a path component. The libfile will be passed directly to
    the dlopen() library call, this implies that normal runtime linking
    conventions apply - i.e. you have three options:

     1. The library name is given with a full path.
     2. The library is in a standard location for shared libraries.
     3. The library is in one of the directories mentioned in the
        LD_LIBRARY_PATH environment variable.

*/

std::vector<std::string>
analysis_config_module_names(const analysis_config_type *config) {
    std::vector<std::string> s;

    for (const auto &analysis_pair : config->analysis_modules)
        s.push_back(analysis_pair.first);

    return s;
}

void analysis_config_load_module(analysis_config_type *config,
                                 analysis_mode_enum mode) {
    analysis_module_type *module = analysis_module_alloc(mode);
    if (module)
        config->analysis_modules[analysis_module_get_name(module)] = module;
    else
        fprintf(stderr, "** Warning: failed to create module \n");
}

void analysis_config_add_module_copy(analysis_config_type *config,
                                     const char *src_name,
                                     const char *target_name) {
    const analysis_module_type *src_module =
        analysis_config_get_module(config, src_name);
    analysis_module_type *target_module = analysis_module_alloc_named(
        analysis_module_get_mode(src_module), target_name);
    config->analysis_modules[target_name] = target_module;
}

analysis_module_type *
analysis_config_get_module(const analysis_config_type *config,
                           const char *module_name) {
    if (analysis_config_has_module(config, module_name)) {
        return config->analysis_modules.at(module_name);
    } else {
        throw exc::invalid_argument("Analysis module named {} not found",
                                    module_name);
    }
}

bool analysis_config_has_module(const analysis_config_type *config,
                                const char *module_name) {
    return (config->analysis_modules.count(module_name) > 0);
}

bool analysis_config_select_module(analysis_config_type *config,
                                   const char *module_name) {
    if (analysis_config_has_module(config, module_name)) {
        analysis_module_type *module =
            analysis_config_get_module(config, module_name);

        config->analysis_module = module;
        return true;
    } else {
        if (config->analysis_module == NULL)
            util_abort("%s: sorry module:%s does not exist - and no module "
                       "currently selected\n",
                       __func__, module_name);
        else
            fprintf(stderr,
                    "** Warning: analysis module:%s does not exist - current "
                    "selection unchanged:%s\n",
                    module_name,
                    analysis_module_get_name(config->analysis_module));
        return false;
    }
}

analysis_module_type *
analysis_config_get_active_module(const analysis_config_type *config) {
    return config->analysis_module;
}

const char *
analysis_config_get_active_module_name(const analysis_config_type *config) {
    if (config->analysis_module)
        return analysis_module_get_name(config->analysis_module);
    else
        return NULL;
}

void analysis_config_load_internal_modules(analysis_config_type *config) {
    analysis_config_load_module(config, ITERATED_ENSEMBLE_SMOOTHER);
    analysis_config_load_module(config, ENSEMBLE_SMOOTHER);
    analysis_config_select_module(config, DEFAULT_ANALYSIS_MODULE);
}

void analysis_config_free(analysis_config_type *config) {
    for (auto &module_pair : config->analysis_modules)
        analysis_module_free(module_pair.second);

    delete config;
}

analysis_config_type *analysis_config_alloc() {
    analysis_config_type *config = new analysis_config_type();

    config->analysis_module = NULL;

    analysis_config_load_internal_modules(config);

    return config;
}
