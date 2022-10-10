#ifndef ERT_ENKF_PLOT_GEN_KW_H
#define ERT_ENKF_PLOT_GEN_KW_H

#include <ert/util/bool_vector.h>
#include <ert/util/stringlist.h>
#include <ert/util/type_macros.h>

#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_plot_gen_kw_vector.hpp>

typedef struct enkf_plot_gen_kw_struct enkf_plot_gen_kw_type;

extern "C" enkf_plot_gen_kw_type *
enkf_plot_gen_kw_alloc(const enkf_config_node_type *enkf_config_node);
extern "C" void enkf_plot_gen_kw_free(enkf_plot_gen_kw_type *gen_kw);
extern "C" enkf_plot_gen_kw_vector_type *
enkf_plot_gen_kw_iget(const enkf_plot_gen_kw_type *vector, int index);
extern "C" void enkf_plot_gen_kw_load(enkf_plot_gen_kw_type *gen_kw,
                                      enkf_fs_type *fs, bool transform_data,
                                      int report_step,
                                      const bool_vector_type *input_mask);

int enkf_plot_gen_kw_get_keyword_index(const enkf_plot_gen_kw_type *gen_kw,
                                       const std::string &keyword);

UTIL_IS_INSTANCE_HEADER(enkf_plot_gen_kw);

#endif
