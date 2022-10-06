#ifndef ERT_ANALYSIS_ITER_CONFIG_H
#define ERT_ANALYSIS_ITER_CONFIG_H

#include <ert/config/config_content.hpp>
#include <ert/config/config_parser.hpp>

typedef struct analysis_iter_config_struct analysis_iter_config_type;

extern "C" void
analysis_iter_config_set_num_iterations(analysis_iter_config_type *config,
                                        int num_iterations);
extern "C" int analysis_iter_config_get_num_iterations(
    const analysis_iter_config_type *config);
void analysis_iter_config_set_num_retries_per_iteration(
    analysis_iter_config_type *config, int num_retries);
extern "C" int analysis_iter_config_get_num_retries_per_iteration(
    const analysis_iter_config_type *config);
extern "C" void
analysis_iter_config_set_case_fmt(analysis_iter_config_type *config,
                                  const char *case_fmt);
extern "C" PY_USED char *
analysis_iter_config_get_case_fmt(analysis_iter_config_type *config);
extern "C" analysis_iter_config_type *analysis_iter_config_alloc();
extern "C" PY_USED analysis_iter_config_type *
analysis_iter_config_alloc_full(const char *case_fmt, int num_iterations,
                                int num_iter_tries);
extern "C" void analysis_iter_config_free(analysis_iter_config_type *config);
const char *analysis_iter_config_iget_case(analysis_iter_config_type *config,
                                           int iter);
void analysis_iter_config_init(analysis_iter_config_type *iter_config,
                               const config_content_type *config);
extern "C" bool
analysis_iter_config_case_fmt_set(const analysis_iter_config_type *config);
extern "C" bool analysis_iter_config_num_iterations_set(
    const analysis_iter_config_type *config);

#endif
