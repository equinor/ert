#ifndef ERT_ENKF_PLOT_GENVECTOR_H
#define ERT_ENKF_PLOT_GENVECTOR_H

#include <ert/util/type_macros.h>

#include <ert/enkf/enkf_config_node.hpp>

typedef struct enkf_plot_genvector_struct enkf_plot_genvector_type;

enkf_plot_genvector_type *
enkf_plot_genvector_alloc(const enkf_config_node_type *enkf_config_node,
                          int iens);
void enkf_plot_genvector_free(enkf_plot_genvector_type *vector);
extern "C" int
enkf_plot_genvector_get_size(const enkf_plot_genvector_type *vector);
void enkf_plot_genvector_load(enkf_plot_genvector_type *vector,
                              enkf_fs_type *fs, int report_step);
extern "C" double
enkf_plot_genvector_iget(const enkf_plot_genvector_type *vector, int index);

UTIL_IS_INSTANCE_HEADER(enkf_plot_genvector);

#endif
