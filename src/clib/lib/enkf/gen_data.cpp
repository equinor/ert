#include <cmath>
#include <filesystem>

#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <ert/util/util.h>

#include <ert/ecl/ecl_sum.h>

#include <ert/logging.hpp>

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_serialize.hpp>
#include <ert/enkf/enkf_util.hpp>
#include <ert/enkf/gen_common.hpp>
#include <ert/enkf/gen_data.hpp>
#include <ert/enkf/gen_data_config.hpp>
#include <ert/enkf/run_arg.hpp>
#include <ert/python.hpp>

namespace fs = std::filesystem;
static auto logger = ert::get_logger("enkf");

/**
   A general data type which can be used to update
   arbitrary data which the EnKF system has *ABSOLUTELY NO IDEA* of
   how is organised; how it should be used in the forward model and so
   on. Similarly to the field objects, the gen_data objects can be
   treated both as parameters and as dynamic data.

   Whether the forward_load function should be called (i.e. it is dynamic
   data) is determined at the enkf_node level, and no busissiness of
   the gen_data implementation.
*/
struct gen_data_struct {
    /** Thin config object - mainly contains filename for remote load */
    gen_data_config_type *config;
    /** Actual storage - will be casted to double or float on use. */
    char *data;
    /** Need this to look up the correct size in the config object. */
    int current_report_step;
    /** Mask of active/not active - loaded from a "_active" file created by the
     * forward model. Not used when used as parameter*/
    bool_vector_type *active_mask;
};

void gen_data_assert_size(gen_data_type *gen_data, int size, int report_step) {
    gen_data_config_assert_size(gen_data->config, size, report_step);
    gen_data->current_report_step = report_step;
}

gen_data_config_type *gen_data_get_config(const gen_data_type *gen_data) {
    return gen_data->config;
}

int gen_data_get_size(const gen_data_type *gen_data) {
    return gen_data_config_get_data_size(gen_data->config,
                                         gen_data->current_report_step);
}

/**
   It is a bug to call this before some function has set the size.
*/
void gen_data_realloc_data(gen_data_type *gen_data) {
    int byte_size = gen_data_config_get_data_size(
                        gen_data->config, gen_data->current_report_step) *
                    sizeof(double);
    gen_data->data = (char *)util_realloc(gen_data->data, byte_size);
}

gen_data_type *gen_data_alloc(const gen_data_config_type *config) {
    gen_data_type *gen_data = (gen_data_type *)util_malloc(sizeof *gen_data);
    gen_data->config = (gen_data_config_type *)config;
    gen_data->data = NULL;
    gen_data->active_mask = bool_vector_alloc(0, true);
    gen_data->current_report_step = -1; /* God - if you ever read this .... */
    return gen_data;
}

void gen_data_copy(const gen_data_type *src, gen_data_type *target) {
    if (src->config == target->config) {
        target->current_report_step = src->current_report_step;

        if (src->data != NULL) {
            size_t byte_size = gen_data_config_get_data_size(
                                   src->config, src->current_report_step) *
                               sizeof(double);
            target->data =
                (char *)util_realloc_copy(target->data, src->data, byte_size);
        }
    } else
        util_abort("%s: do not share config object \n", __func__);
}

void gen_data_free(gen_data_type *gen_data) {
    free(gen_data->data);
    bool_vector_free(gen_data->active_mask);
    free(gen_data);
}

/**
   Observe that this function writes parameter size to disk, that is
   special. The reason is that the config object does not know the
   size (on allocation).

   The function currently writes an empty file (with only a report
   step and a size == 0) in the case where it does not have data. This
   is controlled by the value of the variable write_zero_size; if this
   is changed to false some semantics in the load code must be
   changed.
*/
C_USED bool gen_data_write_to_buffer(const gen_data_type *gen_data,
                                     buffer_type *buffer, int report_step) {
    const bool write_zero_size =
        true; /* true:ALWAYS write a file   false:only write files with size > 0. */
    {
        bool write = write_zero_size;
        int size = gen_data_config_get_data_size(gen_data->config, report_step);
        if (size > 0)
            write = true;

        if (write) {
            size_t byte_size =
                gen_data_config_get_data_size(gen_data->config, report_step) *
                sizeof(double);
            buffer_fwrite_int(buffer, GEN_DATA);
            buffer_fwrite_int(buffer, size);
            buffer_fwrite_int(
                buffer,
                report_step); /* Why the heck do I need to store this ????  It was a mistake ...*/

            buffer_fwrite_compressed(buffer, gen_data->data, byte_size);
            return true;
        } else
            return false; /* When false is returned - the (empty) file will be removed */
    }
}

C_USED void gen_data_read_from_buffer(gen_data_type *gen_data,
                                      buffer_type *buffer, enkf_fs_type *fs,
                                      int report_step) {
    int size;
    enkf_util_assert_buffer_type(buffer, GEN_DATA);
    size = buffer_fread_int(buffer);
    buffer_fskip_int(
        buffer); /* Skipping report_step from the buffer - was a mistake to store it - I think ... */
    {
        size_t byte_size = size * sizeof(double);
        size_t compressed_size = buffer_get_remaining_size(buffer);
        gen_data->data = (char *)util_realloc(gen_data->data, byte_size);
        buffer_fread_compressed(buffer, compressed_size, gen_data->data,
                                byte_size);
    }
    gen_data_assert_size(gen_data, size, report_step);

    if (gen_data_config_is_dynamic(gen_data->config)) {
        gen_data_config_load_active(gen_data->config, fs, report_step, false);
    }
}

void gen_data_serialize(const gen_data_type *gen_data, node_id_type node_id,
                        const ActiveList *active_list, Eigen::MatrixXd &A,
                        int row_offset, int column) {
    const int data_size = gen_data_config_get_data_size(
        gen_data->config, gen_data->current_report_step);

    enkf_matrix_serialize(gen_data->data, data_size, ECL_DOUBLE, active_list, A,
                          row_offset, column);
}

void gen_data_deserialize(gen_data_type *gen_data, node_id_type node_id,
                          const ActiveList *active_list,
                          const Eigen::MatrixXd &A, int row_offset,
                          int column) {
    {
        const int data_size = gen_data_config_get_data_size(
            gen_data->config, gen_data->current_report_step);

        enkf_matrix_deserialize(gen_data->data, data_size, ECL_DOUBLE,
                                active_list, A, row_offset, column);
    }
}

/**
  This function sets the data field of the gen_data instance after the
  data has been loaded from file.
*/
static void gen_data_set_data__(gen_data_type *gen_data, int size,
                                int report_step, const void *data,
                                enkf_fs_type *sim_fs) {
    gen_data_assert_size(gen_data, size, report_step);
    if (gen_data_config_is_dynamic(gen_data->config))
        gen_data_config_update_active(gen_data->config, report_step,
                                      gen_data->active_mask, sim_fs);

    gen_data_realloc_data(gen_data);

    if (size > 0) {
        int byte_size = sizeof(double) * size;

        memcpy(gen_data->data, data, byte_size);
    }
}

static bool gen_data_fload_active__(gen_data_type *gen_data,
                                    const char *filename, int size) {
    /*
     Look for file @filename_active - if that file is found it is
     interpreted as a an active|inactive mask created by the forward
     model.

     The file is assumed to be an ASCII file with integers, 0
     indicates inactive elements and 1 active elements. The file
     should of course be as long as @filename.

     If the file is not found the gen_data->active_mask is set to
     all-true (i.e. the default true value is invoked).
  */
    bool file_exists = false;
    if (gen_data_config_is_dynamic(gen_data->config)) {
        bool_vector_reset(gen_data->active_mask);
        bool_vector_iset(gen_data->active_mask, size - 1, true);
        {
            char *active_file = util_alloc_sprintf("%s_active", filename);
            if (fs::exists(active_file)) {
                file_exists = true;
                FILE *stream = util_fopen(active_file, "r");
                int active_int;
                for (int index = 0; index < size; index++) {
                    if (fscanf(stream, "%d", &active_int) == 1) {
                        if (active_int == 1)
                            bool_vector_iset(gen_data->active_mask, index,
                                             true);
                        else if (active_int == 0)
                            bool_vector_iset(gen_data->active_mask, index,
                                             false);
                        else
                            util_abort("%s: error when loading active mask "
                                       "from:%s only 0 and 1 allowed \n",
                                       __func__, active_file);
                    } else
                        util_abort("%s: error when loading active mask from:%s "
                                   "- file not long enough.\n",
                                   __func__, active_file);
                }
                fclose(stream);
                logger->info("GEN_DATA({}): active information loaded from:{}.",
                             gen_data_get_key(gen_data), active_file);
            } else
                logger->info("GEN_DATA({}): active information not provided.",
                             gen_data_get_key(gen_data));
            free(active_file);
        }
    }
    return file_exists;
}

/**
   This functions loads data from file. Observe that there is *NO*
   header information in this file - the size is determined by seeing
   how much can be successfully loaded.

   The file is loaded with the gen_common_fload_alloc() function.

   When the read is complete it is checked/verified with the config
   object that this file was as long as the others we have loaded for
   other members; it is perfectly OK for the file to not exist. In
   which case a size of zero is set, for this report step.

   Return value is whether file was found or was empty
  - might have to check this in calling scope.
*/
bool gen_data_forward_load(gen_data_type *gen_data, const char *filename,
                           int report_step, enkf_fs_type *fs) {
    bool file_exists = fs::exists(filename);
    if (file_exists) {
        gen_data_file_format_type input_format =
            gen_data_config_get_input_format(gen_data->config);
        auto vec = gen_common_fload_alloc(filename, input_format);
        logger->info("GEN_DATA({}): loading from: {}   size:{}",
                     gen_data_get_key(gen_data), filename, vec.size());
        if (vec.size() > 0) {
            gen_data_fload_active__(gen_data, filename, vec.size());
        } else {
            bool_vector_reset(gen_data->active_mask);
        }
        gen_data_set_data__(gen_data, vec.size(), report_step, vec.data(), fs);
    } else
        logger->warning("GEN_DATA({}): missing file: {}",
                        gen_data_get_key(gen_data), filename);

    return file_exists;
}

static void gen_data_ecl_write_binary(const gen_data_type *gen_data,
                                      const char *file,
                                      ecl_data_type export_type) {
    FILE *stream = util_fopen(file, "w");
    int sizeof_ctype = ecl_type_get_sizeof_ctype(export_type);
    util_fwrite(gen_data->data, sizeof_ctype,
                gen_data_config_get_data_size(gen_data->config,
                                              gen_data->current_report_step),
                stream, __func__);
    fclose(stream);
}

static void gen_data_assert_index(const gen_data_type *gen_data, int index) {
    int current_size = gen_data_config_get_data_size(
        gen_data->config, gen_data->current_report_step);
    if ((index < 0) || (index >= current_size))
        util_abort("%s: index:%d invalid. Valid range: [0,%d) \n", __func__,
                   index, current_size);
}

double gen_data_iget_double(const gen_data_type *gen_data, int index) {
    gen_data_assert_index(gen_data, index);
    double *data = (double *)gen_data->data;
    return data[index];
}

void gen_data_export_data(const gen_data_type *gen_data,
                          double_vector_type *export_data) {
    double_vector_memcpy_from_data(export_data, (const double *)gen_data->data,
                                   gen_data_get_size(gen_data));
}

const char *gen_data_get_key(const gen_data_type *gen_data) {
    return gen_data_config_get_key(gen_data->config);
}

C_USED void gen_data_clear(gen_data_type *gen_data) {
    const int data_size = gen_data_config_get_data_size(
        gen_data->config, gen_data->current_report_step);

    double *data = (double *)gen_data->data;
    for (int i = 0; i < data_size; i++)
        data[i] = 0;
}

double *gen_data_get_double_vector(const gen_data_type *gen_data) {
    double *data = (double *)gen_data->data;
    return data;
}

VOID_ALLOC(gen_data)
VOID_FREE(gen_data)
VOID_READ_FROM_BUFFER(gen_data);
VOID_WRITE_TO_BUFFER(gen_data);
VOID_SERIALIZE(gen_data)
VOID_DESERIALIZE(gen_data)
VOID_CLEAR(gen_data)
