#ifndef ERT_ENKF_PLOT_TVECTOR_H
#define ERT_ENKF_PLOT_TVECTOR_H

#include <stdbool.h>
#include <time.h>

#include <ert/util/util.h>

#include <ert/enkf/enkf_config_node.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_types.hpp>

typedef struct enkf_plot_tvector_struct enkf_plot_tvector_type;

UTIL_SAFE_CAST_HEADER(enkf_plot_tvector);
UTIL_IS_INSTANCE_HEADER(enkf_plot_tvector);

void enkf_plot_tvector_reset(enkf_plot_tvector_type *plot_tvector);
enkf_plot_tvector_type *
enkf_plot_tvector_alloc(const enkf_config_node_type *config_node, int iens);
void enkf_plot_tvector_load(enkf_plot_tvector_type *plot_tvector,
                            enkf_fs_type *fs, const char *user_key);
void enkf_plot_tvector_free(enkf_plot_tvector_type *plot_tvector);
void enkf_plot_tvector_iset(enkf_plot_tvector_type *plot_tvector, int index,
                            time_t time, double value);

extern "C" int
enkf_plot_tvector_size(const enkf_plot_tvector_type *plot_tvector);
extern "C" double
enkf_plot_tvector_iget_value(const enkf_plot_tvector_type *plot_tvector,
                             int index);
extern "C" bool
enkf_plot_tvector_iget_active(const enkf_plot_tvector_type *plot_tvector,
                              int index);
bool enkf_plot_tvector_all_active(const enkf_plot_tvector_type *plot_tvector);

#endif
