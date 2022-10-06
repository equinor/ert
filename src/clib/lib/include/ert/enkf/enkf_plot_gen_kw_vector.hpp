#ifndef ERT_ENKF_PLOT_GEN_KW_VECTOR_H
#define ERT_ENKF_PLOT_GEN_KW_VECTOR_H

#include <ert/util/stringlist.h>
#include <ert/util/type_macros.h>

#include <ert/enkf/enkf_config_node.hpp>

typedef struct enkf_plot_gen_kw_vector_struct enkf_plot_gen_kw_vector_type;

enkf_plot_gen_kw_vector_type *
enkf_plot_gen_kw_vector_alloc(const enkf_config_node_type *config_node,
                              int iens);
void enkf_plot_gen_kw_vector_free(enkf_plot_gen_kw_vector_type *vector);
extern "C" int
enkf_plot_gen_kw_vector_get_size(const enkf_plot_gen_kw_vector_type *vector);
void enkf_plot_gen_kw_vector_reset(enkf_plot_gen_kw_vector_type *vector);
void enkf_plot_gen_kw_vector_load(enkf_plot_gen_kw_vector_type *vector,
                                  enkf_fs_type *fs, bool transform_data,
                                  int report_step);
extern "C" double
enkf_plot_gen_kw_vector_iget(const enkf_plot_gen_kw_vector_type *vector,
                             int index);

UTIL_IS_INSTANCE_HEADER(enkf_plot_gen_kw_vector);

#endif
