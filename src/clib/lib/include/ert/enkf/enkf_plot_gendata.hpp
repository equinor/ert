#ifndef ERT_ENKF_PLOT_GENDATA_H
#define ERT_ENKF_PLOT_GENDATA_H

#include <ert/util/double_vector.h>

#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_plot_genvector.hpp>
#include <ert/enkf/obs_vector.hpp>

typedef struct enkf_plot_gendata_struct enkf_plot_gendata_type;

extern "C" enkf_plot_gendata_type *
enkf_plot_gendata_alloc(const enkf_config_node_type *enkf_config_node);
enkf_plot_gendata_type *
enkf_plot_gendata_alloc_from_obs_vector(const obs_vector_type *obs_vector);
extern "C" void enkf_plot_gendata_free(enkf_plot_gendata_type *data);
extern "C" int enkf_plot_gendata_get_size(const enkf_plot_gendata_type *data);
int enkf_plot_gendata_get_data_size(const enkf_plot_gendata_type *data);
std::vector<bool>
enkf_plot_gendata_active_mask(const enkf_plot_gendata_type *data);
extern "C" enkf_plot_genvector_type *
enkf_plot_gendata_iget(const enkf_plot_gendata_type *plot_data, int index);
extern "C" void enkf_plot_gendata_load(enkf_plot_gendata_type *plot_data,
                                       enkf_fs_type *fs, int report_step);

extern "C" double_vector_type *
enkf_plot_gendata_get_min_values(enkf_plot_gendata_type *plot_data);
extern "C" double_vector_type *
enkf_plot_gendata_get_max_values(enkf_plot_gendata_type *plot_data);

#endif
