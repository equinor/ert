#include <Eigen/Dense>
#include <cmath>
#include <fmt/format.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ert/res_util/subst_list.hpp>
#include <ert/util/buffer.h>
#include <ert/util/util.h>

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/gen_kw.hpp>
#include <ert/enkf/gen_kw_common.hpp>
#include <ert/enkf/gen_kw_config.hpp>
#include <ert/python.hpp>

GET_DATA_SIZE_HEADER(gen_kw);

struct gen_kw_struct {
    const gen_kw_config_type *config;
    double *data;
    subst_list_type *subst_list;
};

void gen_kw_free(gen_kw_type *gen_kw) {
    free(gen_kw->data);
    subst_list_free(gen_kw->subst_list);
    free(gen_kw);
}

extern "C" PY_USED gen_kw_type *gen_kw_alloc(const gen_kw_config_type *config) {
    gen_kw_type *gen_kw = (gen_kw_type *)util_malloc(sizeof *gen_kw);
    gen_kw->config = config;
    gen_kw->subst_list = subst_list_alloc(NULL);
    gen_kw->data = (double *)util_calloc(gen_kw_config_get_data_size(config),
                                         sizeof *gen_kw->data);
    return gen_kw;
}

C_USED void gen_kw_clear(gen_kw_type *gen_kw) {
    int i;
    for (i = 0; i < gen_kw_config_get_data_size(gen_kw->config); i++)
        gen_kw->data[i] = 0.0;
}

int gen_kw_data_size(const gen_kw_type *gen_kw) {
    return gen_kw_config_get_data_size(gen_kw->config);
}

double gen_kw_data_iget(const gen_kw_type *gen_kw, int index,
                        bool do_transform) {
    double value;
    int size = gen_kw_config_get_data_size(gen_kw->config);
    if ((index < 0) || (index >= size))
        util_abort("%s: index:%d invalid. Valid interval: [0,%d>.\n", __func__,
                   index, size);

    if (do_transform) {
        value =
            gen_kw_config_transform(gen_kw->config, index, gen_kw->data[index]);
    } else {
        value = gen_kw->data[index];
    }

    return value;
}

void gen_kw_data_set_vector(gen_kw_type *gen_kw,
                            const double_vector_type *values) {
    int size = gen_kw_config_get_data_size(gen_kw->config);
    if (size == double_vector_size(values)) {
        for (int index = 0; index < size; index++)
            gen_kw->data[index] = double_vector_iget(values, index);
    } else
        util_abort("%s: Invalid size for vector:%d  gen_Kw:%d \n", __func__,
                   double_vector_size(values), size);
}

void gen_kw_data_iset(gen_kw_type *gen_kw, int index, double value) {
    int size = gen_kw_config_get_data_size(gen_kw->config);
    if ((index < 0) || (index >= size))
        util_abort("%s: index:%d invalid. Valid interval: [0,%d>.\n", __func__,
                   index, size);

    gen_kw->data[index] = value;
}

double gen_kw_data_get(gen_kw_type *gen_kw, const char *subkey,
                       bool do_transform) {
    int index = gen_kw_config_get_index(gen_kw->config, subkey);
    return gen_kw_data_iget(gen_kw, index, do_transform);
}

void gen_kw_data_set(gen_kw_type *gen_kw, const char *subkey, double value) {
    int index = gen_kw_config_get_index(gen_kw->config, subkey);
    return gen_kw_data_iset(gen_kw, index, value);
}

bool gen_kw_data_has_key(gen_kw_type *gen_kw, const char *subkey) {
    int index = gen_kw_config_get_index(gen_kw->config, subkey);
    bool has_key =
        ((0 <= index) && (gen_kw_data_size(gen_kw) > index)) ? true : false;
    return has_key;
}

C_USED bool gen_kw_write_to_buffer(const gen_kw_type *gen_kw,
                                   buffer_type *buffer, int report_step) {
    const int data_size = gen_kw_config_get_data_size(gen_kw->config);
    buffer_fwrite_int(buffer, GEN_KW);
    buffer_fwrite(buffer, gen_kw->data, sizeof *gen_kw->data, data_size);
    return true;
}

/*
   As of 17/03/09 (svn 1811) MULTFLT has been depreceated, and GEN_KW
   has been inserted as a 'drop-in-replacement'. This implies that
   existing storage labeled with implemantation type 'MULTFLT' should
   be silently 'upgraded' to 'GEN_KW'.
*/

#define MULTFLT 102
void gen_kw_read_from_buffer(gen_kw_type *gen_kw, buffer_type *buffer,
                             enkf_fs_type *fs, int report_step) {
    const int data_size = gen_kw_config_get_data_size(gen_kw->config);
    ert_impl_type file_type;
    file_type = (ert_impl_type)buffer_fread_int(buffer);
    if ((file_type == GEN_KW) || (file_type == MULTFLT)) {
        size_t expected_size =
            buffer_get_remaining_size(buffer) / sizeof *gen_kw->data;
        if (expected_size != data_size) {
            const char *key = gen_kw_config_get_key(gen_kw->config);
            throw std::range_error(
                fmt::format("The configuration of GEN_KW parameter {} is of "
                            "size {}, expected {}",
                            key, data_size, expected_size));
        }
        buffer_fread(buffer, gen_kw->data, sizeof *gen_kw->data, data_size);
    }
}
#undef MULTFLT

void gen_kw_serialize(const gen_kw_type *gen_kw, node_id_type node_id,
                      const ActiveList *active_list, Eigen::MatrixXd &A,
                      int row_offset, int column) {
    const int data_size = gen_kw_config_get_data_size(gen_kw->config);
    enkf_matrix_serialize(gen_kw->data, data_size, ECL_DOUBLE, active_list, A,
                          row_offset, column);
}

void gen_kw_deserialize(gen_kw_type *gen_kw, node_id_type node_id,
                        const ActiveList *active_list, const Eigen::MatrixXd &A,
                        int row_offset, int column) {
    const int data_size = gen_kw_config_get_data_size(gen_kw->config);
    enkf_matrix_deserialize(gen_kw->data, data_size, ECL_DOUBLE, active_list, A,
                            row_offset, column);
}

void gen_kw_filter_file(const gen_kw_type *gen_kw, const char *target_file) {
    const char *template_file = gen_kw_config_get_template_file(gen_kw->config);
    if (template_file != NULL) {
        const int size = gen_kw_config_get_data_size(gen_kw->config);
        int ikw;

        for (ikw = 0; ikw < size; ikw++) {
            const char *key =
                gen_kw_config_get_tagged_name(gen_kw->config, ikw);
            subst_list_append_owned_ref(
                gen_kw->subst_list, key,
                util_alloc_sprintf("%g",
                                   gen_kw_config_transform(gen_kw->config, ikw,
                                                           gen_kw->data[ikw])),
                NULL);
        }

        /*
      If the target_file already exists as a symbolic link the
      symbolic link is removed before creating the target file. The is
      to ensure against existing symlinks pointing to a common file
      outside the realization root.
    */
        if (util_is_link(target_file))
            remove(target_file);

        subst_list_filter_file(gen_kw->subst_list, template_file, target_file);
    } else
        util_abort("%s: internal error - tried to filter gen_kw instance "
                   "without template file.\n",
                   __func__);
}

const char *gen_kw_get_name(const gen_kw_type *gen_kw, int kw_nr) {
    return gen_kw_config_iget_name(gen_kw->config, kw_nr);
}

/**
   Will return 0.0 on invalid input, and set valid -> false. It is the
   responsibility of the calling scope to check valid.
*/
C_USED bool gen_kw_user_get(const gen_kw_type *gen_kw, const char *key,
                            int report_step, double *value) {
    int index = gen_kw_config_get_index(gen_kw->config, key);

    if (index >= 0) {
        *value =
            gen_kw_config_transform(gen_kw->config, index, gen_kw->data[index]);
        return true;
    } else {
        *value = 0.0;
        fprintf(stderr,
                "** Warning:could not lookup key:%s in gen_kw instance \n",
                key);
        return false;
    }
}

VOID_ALLOC(gen_kw);
VOID_FREE(gen_kw)
VOID_USER_GET(gen_kw)
VOID_WRITE_TO_BUFFER(gen_kw)
VOID_READ_FROM_BUFFER(gen_kw)
VOID_SERIALIZE(gen_kw)
VOID_DESERIALIZE(gen_kw)
VOID_CLEAR(gen_kw)

namespace {
void gen_kw_export_values(const gen_kw_type *gen_kw, py::dict exports) {
    auto size = gen_kw_config_get_data_size(gen_kw->config);

    for (int ikw{}; ikw < size; ++ikw) {
        auto key = gen_kw_config_get_key(gen_kw->config);
        auto parameter = gen_kw_config_iget_name(gen_kw->config, ikw);

        auto value =
            gen_kw_config_transform(gen_kw->config, ikw, gen_kw->data[ikw]);

        if (!exports.contains(key))
            exports[key] = py::dict{};
        exports[key][parameter] = value;

        if (gen_kw_config_should_use_log_scale(gen_kw->config, ikw)) {
            auto log_key = fmt::format("LOG10_{}", key);

            if (!exports.contains(log_key.c_str()))
                exports[log_key.c_str()] = py::dict{};
            exports[log_key.c_str()][parameter] = log10(value);
        }
    }
}

void gen_kw_ecl_write(const gen_kw_type *gen_kw, const char *run_path,
                      const char *base_file, py::dict exports) {
    char *target_file;
    if (run_path)
        target_file = util_alloc_filename(run_path, base_file, NULL);
    else
        target_file = util_alloc_string_copy(base_file);

    gen_kw_filter_file(gen_kw, target_file);
    free(target_file);

    gen_kw_export_values(gen_kw, exports);
}
} // namespace

ERT_CLIB_SUBMODULE("gen_kw", m) {
    m.def("generate_parameter_file",
          [](Cwrap<enkf_node_type> enkf_node, const std::string &run_path,
             const std::optional<std::string> &opt_file, py::dict exports) {
              if (enkf_node_get_impl_type(enkf_node) != GEN_KW)
                  throw py::value_error{"EnkfNode must be of type GEN_KW"};

              auto file = opt_file ? opt_file->c_str() : nullptr;
              gen_kw_ecl_write(
                  static_cast<gen_kw_type *>(enkf_node_value_ptr(enkf_node)),
                  run_path.c_str(), file, exports);
          });
}
