#ifndef ERT_ENKF_PLOT_DATA_H
#define ERT_ENKF_PLOT_DATA_H

#include <stdbool.h>

#include <ert/util/bool_vector.h>
#include <ert/util/type_macros.h>

#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_plot_tvector.hpp>
#include <ert/enkf/enkf_types.hpp>

typedef struct enkf_plot_data_struct enkf_plot_data_type;

extern "C" enkf_plot_data_type *
enkf_plot_data_alloc(const enkf_config_node_type *config_node);
extern "C" void enkf_plot_data_free(enkf_plot_data_type *plot_data);
extern "C" void enkf_plot_data_load(enkf_plot_data_type *plot_data,
                                    enkf_fs_type *fs, const char *user_key);
extern "C" int enkf_plot_data_get_size(const enkf_plot_data_type *plot_data);
extern "C" enkf_plot_tvector_type *
enkf_plot_data_iget(const enkf_plot_data_type *plot_data, int index);

UTIL_IS_INSTANCE_HEADER(enkf_plot_data);

#endif
