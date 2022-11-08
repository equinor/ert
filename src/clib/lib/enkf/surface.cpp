#include <Eigen/Dense>
#include <cmath>
#include <ert/util/util.h>
#include <stdlib.h>

#include <ert/enkf/enkf_node.hpp>
#include <ert/geometry/geo_surface.h>

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/surface.hpp>
#include <ert/python.hpp>

struct surface_struct {
    /** Can not be NULL - var_type is set on first load. */
    surface_config_type *config;
    /** Size should always be one */
    double *data;
};

bool surface_fload(surface_type *surface, const char *filename) {
    bool ret = false;
    if (filename) {
        const geo_surface_type *base_surface =
            surface_config_get_base_surface(surface->config);
        ret = geo_surface_fload_irap_zcoord(base_surface, filename,
                                            surface->data);
    }
    return ret;
}

bool surface_initialize(surface_type *surface, int iens, const char *filename) {
    return surface_fload(surface, filename);
}

surface_type *surface_alloc(const surface_config_type *surface_config) {
    surface_type *surface = (surface_type *)util_malloc(sizeof *surface);
    surface->config = (surface_config_type *)surface_config;
    {
        const int data_size = surface_config_get_data_size(surface_config);
        surface->data = (double *)util_calloc(data_size, sizeof *surface->data);
    }
    return surface;
}

void surface_read_from_buffer(surface_type *surface, buffer_type *buffer,
                              enkf_fs_type *fs, int report_step) {
    int size = surface_config_get_data_size(surface->config);
    enkf_util_assert_buffer_type(buffer, SURFACE);
    buffer_fread(buffer, surface->data, sizeof *surface->data, size);
}

bool surface_write_to_buffer(const surface_type *surface, buffer_type *buffer,
                             int report_step) {
    int size = surface_config_get_data_size(surface->config);
    buffer_fwrite_int(buffer, SURFACE);
    buffer_fwrite(buffer, surface->data, sizeof *surface->data, size);
    return true;
}

void surface_free(surface_type *surface) {
    free(surface->data);
    free(surface);
}

void surface_serialize(const surface_type *surface, node_id_type node_id,
                       const ActiveList *active_list, Eigen::MatrixXd &A,
                       int row_offset, int column) {
    const surface_config_type *config = surface->config;
    const int data_size = surface_config_get_data_size(config);

    enkf_matrix_serialize(surface->data, data_size, ECL_DOUBLE, active_list, A,
                          row_offset, column);
}

void surface_deserialize(surface_type *surface, node_id_type node_id,
                         const ActiveList *active_list,
                         const Eigen::MatrixXd &A, int row_offset, int column) {
    const surface_config_type *config = surface->config;
    const int data_size = surface_config_get_data_size(config);

    enkf_matrix_deserialize(surface->data, data_size, ECL_DOUBLE, active_list,
                            A, row_offset, column);
}

void surface_ecl_write(const surface_type *surface, const char *run_path,
                       const char *base_file) {
    char *target_file = util_alloc_filename(run_path, base_file, NULL);
    surface_config_ecl_write(surface->config, target_file, surface->data);
    free(target_file);
}

VOID_ALLOC(surface)
VOID_FREE(surface)
VOID_WRITE_TO_BUFFER(surface)
VOID_READ_FROM_BUFFER(surface)
VOID_SERIALIZE(surface)
VOID_DESERIALIZE(surface)
VOID_INITIALIZE(surface)

ERT_CLIB_SUBMODULE("surface", m) {
    m.def("generate_parameter_file",
          [](Cwrap<enkf_node_type> enkf_node, const std::string &run_path,
             const std::optional<std::string> &opt_file) {
              if (enkf_node_get_impl_type(enkf_node) != SURFACE)
                  throw py::value_error{"EnkfNode must be of type SURFACE"};

              auto file = opt_file ? opt_file->c_str() : nullptr;
              surface_ecl_write(
                  static_cast<surface_type *>(enkf_node_value_ptr(enkf_node)),
                  run_path.c_str(), file);
          });
}
