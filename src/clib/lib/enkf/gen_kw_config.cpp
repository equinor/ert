#include <filesystem>
#include <numeric>

#include <stdlib.h>
#include <string.h>

#include <ert/python.hpp>
#include <ert/util/hash.h>
#include <ert/util/util.h>
#include <ert/util/vector.h>

#include <ert/config/config_parser.hpp>
#include <ert/logging.hpp>

#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/gen_kw_config.hpp>
#include <ert/enkf/trans_func.hpp>

namespace fs = std::filesystem;
static auto logger = ert::get_logger("gen_kw_config");

typedef struct {
    char *name;
    trans_func_type *trans_func;
} gen_kw_parameter_type;

struct gen_kw_config_struct {
    /** Vector of gen_kw_parameter_type instances. */
    vector_type *parameters;
};

static gen_kw_parameter_type *
gen_kw_parameter_alloc(const char *parameter_name) {
    gen_kw_parameter_type *parameter =
        (gen_kw_parameter_type *)util_malloc(sizeof *parameter);
    parameter->name = util_alloc_string_copy(parameter_name);
    parameter->trans_func = NULL;
    return parameter;
}

static void gen_kw_parameter_free(gen_kw_parameter_type *parameter) {
    free(parameter->name);
    if (parameter->trans_func != NULL)
        trans_func_free(parameter->trans_func);
    free(parameter);
}

static void gen_kw_parameter_free__(void *__parameter) {
    auto parameter = static_cast<gen_kw_parameter_type *>(__parameter);
    gen_kw_parameter_free(parameter);
}

static void gen_kw_parameter_set_trans_func(gen_kw_parameter_type *parameter,
                                            trans_func_type *trans_func) {
    if (parameter->trans_func != NULL)
        trans_func_free(parameter->trans_func);
    parameter->trans_func = trans_func;
}

void gen_kw_config_set_parameter_file(gen_kw_config_type *config,
                                      const char *parameter_file) {
    vector_clear(config->parameters);
    if (parameter_file != NULL) {
        config_parser_type *parser = config_alloc();
        config_content_type *content =
            config_parse(parser, parameter_file, "--", NULL, NULL, NULL,
                         CONFIG_UNRECOGNIZED_ADD, false);
        if (!content->valid) {
            auto header = fmt::format(
                "encountered errors while parsing GEN_KW parameter file {}",
                parameter_file);
            std::string errors;
            for (auto &error : content->parse_errors) {
                errors += error;
            }
            logger->warning("{}\n{}", header, errors);
        }
        for (auto parse_error : content->parse_errors) {
            logger->warning(parse_error);
        }
        for (int item_index = 0; item_index < config_content_get_size(content);
             item_index++) {
            const config_content_node_type *node =
                config_content_iget_node(content, item_index);
            const char *parameter_name = config_content_node_get_kw(node);
            gen_kw_parameter_type *parameter =
                gen_kw_parameter_alloc(parameter_name);
            trans_func_type *trans_func =
                trans_func_alloc(config_content_node_get_stringlist(node));
            if (trans_func) {
                gen_kw_parameter_set_trans_func(parameter, trans_func);
                vector_append_owned_ref(config->parameters, parameter,
                                        gen_kw_parameter_free__);
            } else
                util_abort(
                    "%s: failed to create tranformation function for %s\n",
                    __func__, parameter_name);
        }
        config_content_free(content);
        config_free(parser);
    }
}

gen_kw_config_type *gen_kw_config_alloc_empty() {
    gen_kw_config_type *gen_kw_config =
        (gen_kw_config_type *)util_malloc(sizeof *gen_kw_config);

    gen_kw_config->parameters = vector_alloc_new();
    return gen_kw_config;
}

double gen_kw_config_transform(const gen_kw_config_type *config, int index,
                               double x) {
    const gen_kw_parameter_type *parameter =
        (const gen_kw_parameter_type *)vector_iget_const(config->parameters,
                                                         index);
    return trans_func_eval(parameter->trans_func, x);
}

void gen_kw_config_free(gen_kw_config_type *gen_kw_config) {
    vector_free(gen_kw_config->parameters);
    free(gen_kw_config);
}

int gen_kw_config_get_data_size(const gen_kw_config_type *gen_kw_config) {
    return vector_get_size(gen_kw_config->parameters);
}

const char *gen_kw_config_iget_name(const gen_kw_config_type *config,
                                    int kw_nr) {
    const gen_kw_parameter_type *parameter =
        (const gen_kw_parameter_type *)vector_iget(config->parameters, kw_nr);
    return parameter->name;
}

VOID_FREE(gen_kw_config)
VOID_GET_DATA_SIZE(gen_kw)

ERT_CLIB_SUBMODULE("gen_kw_config", m) {
    m.def(
        "get_function_parameter_values",
        [](Cwrap<gen_kw_config_type> self, int index) {
            const gen_kw_parameter_type *parameter =
                (const gen_kw_parameter_type *)vector_iget_const(
                    self->parameters, index);
            return trans_func_get_params(parameter->trans_func);
        },
        py::arg("self"), py::arg("index"));
}
