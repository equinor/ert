#include "ert/python.hpp"
#include <ert/concurrency.hpp>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <ert/util/util.h>

#include <ert/logging.hpp>
#include <ert/res_util/file_utils.hpp>
#include <ert/res_util/path_fmt.hpp>
#include <ert/res_util/string.hpp>

#include <ert/enkf/block_fs_driver.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_state.hpp>
#include <ert/enkf/ensemble_config.hpp>

namespace fs = std::filesystem;
static auto logger = ert::get_logger("enkf");

/*
  The interface
  -------------

  The unit of storage in the enkf_fs system is one enkf_node instance. The
  interface between the storage system and the rest of the EnKF system is
  through the enkf_fs functions:

    enkf_fs_fread_node()
    enkf_fs_has_node()
    enkf_fs_fwrite_node()


  So all these functions (partly except enkf_fs_has_node()) work on a enkf_node
  instance, and in addition they take the following input:

    - iens        : ensemble member number
    - report_step : the report_step number we are interested in
    - state       : whether we are considering an analyzed node or a forecast.

  The drivers
  -----------

  The enkf_fs layer does not self implement the functions to read and write
  nodes. Instead what happens is:

    1. We determine the type of the node (static/dynamic/parameter), and select
       the appropriate driver.

    2. The appropriate driver is called to implement e.g. the fread_node
       functions.

  The different types of data have different characteristics, which the driver is
  implemented to support. The characteristics the drivers support are the
  following:


  Mounting the filesystem
  -----------------------

  The important point is that the moment ensemble information 
  has hit the filesystem later versions of the enkf program must 
  support exactly that lay-out, those drivers+++.
  To ensure this I see two possibilities:

    1. We can freeze the filesystem drivers, and the layout on disk
       indefinitely.

    2. We can store the information needed to bootstrap the drivers,
       according to the current layout on disk, in the
       filesystem. I.e. something like a '/etc/fstab' file.

  We have chosen the second alternative. Currently this implemented as
  follows:

    1. In main() we query for the file {root-path}/enkf_mount_info. If
       that file does not exists it is created by calls to the
       selected drivers xxxx_fwrite_mount_info() functions.

    2. enkf_fs_mount() is called with the enkf_mount_info as input.

  The enkf_mount_info file (BINARY) consists of four records (one for
  each driver, including the index). The format of each record is:

     DRIVER_CATEGORY   DRIVER_ID    INFO
     int               int          void *

  The driver category should be one of the four integer values in
  ert::block_fs_driver (fs_types.hpp) and DRIVER_ID is one of the integer
  values in fs_driver_impl. The last void * data is whatever
  (serialized) info the driver needs to bootstrap. This info is
  written by the drivers xxxx_fwrite_mount_info() function, and it is
  used when the driver is allocated with xxxx_fread_alloc().

  The different drivers can be in arbitrary order in the
  enkf_mount_info file, but when four records are read it checks that
  all drivers have been initialized, and aborts if that is not the
  case.

  If the enkf_mount_info file is deleted that can cause problems.
  It is currently 'protected' with chomd a-w - but that is of course not
  foolprof.
*/

#define ENKF_MOUNT_MAP "enkf_mount_info"
#define SUMMARY_KEY_SET_FILE "summary-key-set"
#define TIME_MAP_FILE "time-map"
#define STATE_MAP_FILE "state-map"
#define CASE_CONFIG_FILE "case_config"

struct enkf_fs_struct {
    std::string case_name;
    char *mount_point;

    std::unique_ptr<ert::block_fs_driver> dynamic_forecast;
    std::unique_ptr<ert::block_fs_driver> parameter;
    std::unique_ptr<ert::block_fs_driver> index;

    /** Whether this filesystem has been mounted read-only. */
    bool read_only;
    std::shared_ptr<TimeMap> time_map;
    std::shared_ptr<StateMap> state_map;
    /* The variables below here are for storing arbitrary files within the
     * enkf_fs storage directory, but not as serialized enkf_nodes. */
    path_fmt_type *case_fmt;
    path_fmt_type *case_member_fmt;
    path_fmt_type *case_tstep_fmt;
    path_fmt_type *case_tstep_member_fmt;
};

enkf_fs_type *enkf_fs_get_ref(enkf_fs_type *fs) { return fs; }

enkf_fs_type *enkf_fs_alloc_empty(const char *mount_point,
                                  unsigned ensemble_size, bool read_only) {
    enkf_fs_type *fs = new enkf_fs_type;
    fs->time_map = std::make_shared<TimeMap>();
    fs->state_map = std::make_shared<StateMap>(ensemble_size);
    fs->read_only = read_only;
    fs->mount_point = strdup(mount_point);
    auto mount_path = fs::path(mount_point);
    std::string case_name = mount_path.filename();

    return fs;
}

void enkf_fs_init_path_fmt(enkf_fs_type *fs) {
    /*
    Installing the path_fmt instances for the storage of arbitrary files.
  */
    fs->case_fmt = path_fmt_alloc_directory_fmt(DEFAULT_CASE_PATH);
    fs->case_member_fmt =
        path_fmt_alloc_directory_fmt(DEFAULT_CASE_MEMBER_PATH);
    fs->case_tstep_fmt = path_fmt_alloc_directory_fmt(DEFAULT_CASE_TSTEP_PATH);
    fs->case_tstep_member_fmt =
        path_fmt_alloc_directory_fmt(DEFAULT_CASE_TSTEP_MEMBER_PATH);
}

static void enkf_fs_create_block_fs(FILE *stream, int num_drivers,
                                    const char *mount_point) {

    block_fs_driver_create_fs(stream, mount_point, DRIVER_PARAMETER,
                              num_drivers, "Ensemble/mod_%d", "PARAMETER");
    block_fs_driver_create_fs(stream, mount_point, DRIVER_DYNAMIC_FORECAST,
                              num_drivers, "Ensemble/mod_%d", "FORECAST");
    block_fs_driver_create_fs(stream, mount_point, DRIVER_INDEX, 1, "Index",
                              "INDEX");
}

static void enkf_fs_assign_driver(enkf_fs_type *fs,
                                  ert::block_fs_driver *driver,
                                  fs_driver_enum driver_type) {
    switch (driver_type) {
    case (DRIVER_PARAMETER):
        fs->parameter.reset(driver);
        break;
    case (DRIVER_DYNAMIC_FORECAST):
        fs->dynamic_forecast.reset(driver);
        break;
    case (DRIVER_INDEX):
        fs->index.reset(driver);
        break;
    }
}

static enkf_fs_type *enkf_fs_mount_block_fs(FILE *fstab_stream,
                                            const char *mount_point,
                                            unsigned ensemble_size,
                                            bool read_only) {
    enkf_fs_type *fs =
        enkf_fs_alloc_empty(mount_point, ensemble_size, read_only);

    {
        while (true) {
            fs_driver_enum driver_type;
            if (fread(&driver_type, sizeof driver_type, 1, fstab_stream) == 1) {
                if (fs_types_valid(driver_type)) {
                    ert::block_fs_driver *driver = ert::block_fs_driver::open(
                        fstab_stream, mount_point, fs->read_only);
                    enkf_fs_assign_driver(fs, driver, driver_type);
                } else
                    block_fs_driver_fskip(fstab_stream);
            } else
                break;
        }
    }

    return fs;
}

enkf_fs_type *enkf_fs_create_fs(const char *mount_point,
                                fs_driver_impl driver_id,
                                unsigned ensemble_size, bool mount) {
    /*
	 * NOTE: This value is the (maximum) number of concurrent files
	 * used by ert::block_fs_driver -objects. These objects will
	 * occasionally schedule one std::future for each file, hence
	 * this is sometimes the number of concurrently executing futures.
	 * (In other words - don't set it to 100000...)
	 */
    const int num_drivers = 32;

    FILE *stream = fs_driver_open_fstab(mount_point, true);
    if (stream != NULL) {
        fs_driver_init_fstab(stream, driver_id);
        {
            switch (driver_id) {
            case (BLOCK_FS_DRIVER_ID):
                enkf_fs_create_block_fs(stream, num_drivers, mount_point);
                break;
            default:
                util_abort("%s: Invalid driver_id value:%d \n", __func__,
                           driver_id);
            }
        }
        fclose(stream);
    }

    if (mount)
        return enkf_fs_mount(mount_point, ensemble_size);
    else
        return NULL;
}

static void enkf_fs_fsync_time_map(enkf_fs_type *fs) {
    char *filename = enkf_fs_alloc_case_filename(fs, TIME_MAP_FILE);
    fs->time_map->write_binary(filename);
    free(filename);
}

static void enkf_fs_fread_time_map(enkf_fs_type *fs) {
    char *filename = enkf_fs_alloc_case_filename(fs, TIME_MAP_FILE);
    fs->time_map->read_binary(filename);
    free(filename);
}

static void enkf_fs_fsync_state_map(enkf_fs_type *fs) {
    char *filename = enkf_fs_alloc_case_filename(fs, STATE_MAP_FILE);
    try {
        fs->state_map->write(filename);
    } catch (std::ios_base::failure &) {
        // Write errors are ignored
    }
    free(filename);
}

static void enkf_fs_fread_state_map(enkf_fs_type *fs) {
    char *filename = enkf_fs_alloc_case_filename(fs, STATE_MAP_FILE);
    try {
        fs->state_map->read(filename);
    } catch (const std::ios_base::failure &) {
        /* Read error is ignored. StateMap is reset */
    }
    free(filename);
}

enkf_fs_type *enkf_fs_mount(const char *mount_point, unsigned ensemble_size,
                            bool read_only) {
    FILE *stream = fs_driver_open_fstab(mount_point, false);

    if (!stream)
        return NULL;

    enkf_fs_type *fs = NULL;
    fs_driver_assert_magic(stream);
    fs_driver_assert_version(stream, mount_point);

    fs_driver_impl driver_id = (fs_driver_impl)util_fread_int(stream);

    switch (driver_id) {
    case (BLOCK_FS_DRIVER_ID):
        fs = enkf_fs_mount_block_fs(stream, mount_point, ensemble_size,
                                    read_only);
        logger->debug("Mounting (block_fs) point {}.", mount_point);
        break;
    default:
        util_abort("%s: unrecognized driver_id:%d \n", __func__, driver_id);
    }

    fclose(stream);
    enkf_fs_init_path_fmt(fs);
    enkf_fs_fread_time_map(fs);
    enkf_fs_fread_state_map(fs);

    enkf_fs_get_ref(fs);
    return fs;
}

bool enkf_fs_exists(const char *mount_point) {
    bool exists = false;

    FILE *stream = fs_driver_open_fstab(mount_point, false);
    if (stream != NULL) {
        exists = true;
        fclose(stream);
    }

    return exists;
}

void enkf_fs_sync(enkf_fs_type *fs) {
    if (!fs->read_only) {
        enkf_fs_fsync(fs);
    }
}

void enkf_fs_umount(enkf_fs_type *fs) {
    free(fs->mount_point);
    path_fmt_free(fs->case_fmt);
    path_fmt_free(fs->case_member_fmt);
    path_fmt_free(fs->case_tstep_fmt);
    path_fmt_free(fs->case_tstep_member_fmt);

    delete fs;
}

static ert::block_fs_driver *enkf_fs_select_driver(enkf_fs_type *fs,
                                                   enkf_var_type var_type,
                                                   const char *key) {
    switch (var_type) {
    case (DYNAMIC_RESULT):
        return fs->dynamic_forecast.get();
    case (EXT_PARAMETER):
        return fs->parameter.get();
    case (PARAMETER):
        return fs->parameter.get();
    default:
        util_abort("%s: fatal internal error - could not determine enkf_fs "
                   "driver for object:%s[integer type:%d] - aborting.\n",
                   __func__, key, var_type);
    }
    std::abort();
}

void enkf_fs_fsync(enkf_fs_type *fs) {
    fs->parameter->fsync();
    fs->dynamic_forecast->fsync();
    fs->index->fsync();

    enkf_fs_fsync_time_map(fs);
    enkf_fs_fsync_state_map(fs);
}

void enkf_fs_fread_node(enkf_fs_type *enkf_fs, buffer_type *buffer,
                        const char *node_key, enkf_var_type var_type,
                        int report_step, int iens) {

    ert::block_fs_driver *driver =
        (ert::block_fs_driver *)enkf_fs_select_driver(enkf_fs, var_type,
                                                      node_key);
    if (var_type == PARAMETER)
        /* Parameters are *ONLY* stored at report_step == 0 */
        report_step = 0;

    buffer_rewind(buffer);
    driver->load_node(node_key, report_step, iens, buffer);
}

void enkf_fs_fread_vector(enkf_fs_type *enkf_fs, buffer_type *buffer,
                          const char *node_key, enkf_var_type var_type,
                          int iens) {

    ert::block_fs_driver *driver =
        (ert::block_fs_driver *)enkf_fs_select_driver(enkf_fs, var_type,
                                                      node_key);

    buffer_rewind(buffer);
    driver->load_vector(node_key, iens, buffer);
}

bool enkf_fs_has_node(enkf_fs_type *enkf_fs, const char *node_key,
                      enkf_var_type var_type, int report_step, int iens) {
    ert::block_fs_driver *driver =
        enkf_fs_select_driver(enkf_fs, var_type, node_key);
    return driver->has_node(node_key, report_step, iens);
}

bool enkf_fs_has_vector(enkf_fs_type *enkf_fs, const char *node_key,
                        enkf_var_type var_type, int iens) {
    ert::block_fs_driver *driver =
        enkf_fs_select_driver(enkf_fs, var_type, node_key);
    return driver->has_vector(node_key, iens);
}

void enkf_fs_fwrite_node(enkf_fs_type *enkf_fs, buffer_type *buffer,
                         const char *node_key, enkf_var_type var_type,
                         int report_step, int iens) {
    if (enkf_fs->read_only)
        util_abort("%s: attempt to write to read_only filesystem mounted at:%s "
                   "- aborting. \n",
                   __func__, enkf_fs->mount_point);

    if ((var_type == PARAMETER) && (report_step > 0))
        util_abort(
            "%s: Parameters can only be saved for report_step = 0   %s:%d\n",
            __func__, node_key, report_step);
    ert::block_fs_driver *driver =
        enkf_fs_select_driver(enkf_fs, var_type, node_key);
    driver->save_node(node_key, report_step, iens, buffer);
}

void enkf_fs_fwrite_vector(enkf_fs_type *enkf_fs, buffer_type *buffer,
                           const char *node_key, enkf_var_type var_type,
                           int iens) {
    if (enkf_fs->read_only)
        util_abort("%s: attempt to write to read_only filesystem mounted at:%s "
                   "- aborting. \n",
                   __func__, enkf_fs->mount_point);
    ert::block_fs_driver *driver =
        enkf_fs_select_driver(enkf_fs, var_type, node_key);
    driver->save_vector(node_key, iens, buffer);
}

const char *enkf_fs_get_mount_point(const enkf_fs_type *fs) {
    return fs->mount_point;
}

bool enkf_fs_is_read_only(const enkf_fs_type *fs) { return fs->read_only; }

void enkf_fs_set_read_only(enkf_fs_type *fs, bool read_only) {
    fs->read_only = read_only;
}

char *enkf_fs_alloc_case_filename(const enkf_fs_type *fs,
                                  const char *input_name) {
    char *filename =
        path_fmt_alloc_file(fs->case_fmt, false, fs->mount_point, input_name);
    return filename;
}

char *enkf_fs_alloc_case_tstep_filename(const enkf_fs_type *fs, int tstep,
                                        const char *input_name) {
    char *filename = path_fmt_alloc_file(fs->case_tstep_fmt, false,
                                         fs->mount_point, tstep, input_name);
    return filename;
}

char *enkf_fs_alloc_case_tstep_member_filename(const enkf_fs_type *fs,
                                               int tstep, int iens,
                                               const char *input_name) {
    char *filename =
        path_fmt_alloc_file(fs->case_tstep_member_fmt, false, fs->mount_point,
                            tstep, iens, input_name);
    return filename;
}

FILE *enkf_fs_open_case_tstep_file(const enkf_fs_type *fs,
                                   const char *input_name, int tstep,
                                   const char *mode) {
    char *filename = enkf_fs_alloc_case_tstep_filename(fs, tstep, input_name);
    auto stream = mkdir_fopen(fs::path(filename), mode);
    free(filename);
    return stream;
}

static FILE *enkf_fs_open_exfile(const char *filename) {
    if (fs::exists(filename))
        return util_fopen(filename, "r");
    else
        return NULL;
}

FILE *enkf_fs_open_excase_file(const enkf_fs_type *fs, const char *input_name) {
    char *filename = enkf_fs_alloc_case_filename(fs, input_name);
    FILE *stream = enkf_fs_open_exfile(filename);
    free(filename);
    return stream;
}

FILE *enkf_fs_open_excase_tstep_file(const enkf_fs_type *fs,
                                     const char *input_name, int tstep) {
    char *filename = enkf_fs_alloc_case_tstep_filename(fs, tstep, input_name);
    FILE *stream = enkf_fs_open_exfile(filename);
    free(filename);
    return stream;
}

TimeMap &enkf_fs_get_time_map(const enkf_fs_type *fs) { return *fs->time_map; }

StateMap &enkf_fs_get_state_map(enkf_fs_type *fs) { return *fs->state_map; }

ERT_CLIB_SUBMODULE("enkf_fs", m) {
    using namespace py::literals;

    m.def(
        "get_state_map",
        [](Cwrap<enkf_fs_type> self) { return self->state_map; }, "self"_a);
    m.def(
        "get_time_map", [](Cwrap<enkf_fs_type> self) { return self->time_map; },
        "self"_a);
    m.def(
        "is_initialized",
        [](Cwrap<enkf_fs_type> fs, Cwrap<ensemble_config_type> ensemble_config,
           std::vector<std::string> parameter_keys, int ens_size) {
            bool initialized = true;
            for (int ikey = 0; (ikey < parameter_keys.size()) && initialized;
                 ikey++) {
                const enkf_config_node_type *config_node =
                    ensemble_config_get_node(ensemble_config,
                                             parameter_keys[ikey].c_str());

                initialized = enkf_fs_has_node(fs, config_node->key,
                                               config_node->var_type, 0, 0);
                for (int iens = 0; (iens < ens_size) && initialized; iens++) {
                    initialized = enkf_fs_has_node(
                        fs, config_node->key, config_node->var_type, 0, iens);
                }
            }
            return initialized;
        },
        py::arg("self"), py::arg("ensemble_config"), py::arg("parameter_names"),
        py::arg("ensemble_size"));
    m.def(
        "copy_from_case",
        [](Cwrap<enkf_fs_type> source_case,
           Cwrap<ensemble_config_type> ensemble_config,
           Cwrap<enkf_fs_type> target_case, int report_step,
           std::vector<std::string> &node_list, std::vector<bool> &iactive) {
            auto &target_state_map = enkf_fs_get_state_map(target_case);

            for (auto &node : node_list) {
                enkf_config_node_type *config_node =
                    ensemble_config_get_node(ensemble_config, node.c_str());

                int src_iens = 0;
                for (auto mask : iactive) {
                    if (mask) {
                        node_id_type src_id = {.report_step = report_step,
                                               .iens = src_iens};
                        node_id_type target_id = {.report_step = 0,
                                                  .iens = src_iens};

                        /* The copy is careful ... */
                        if (enkf_fs_has_node(source_case, config_node->key,
                                             config_node->var_type, report_step,
                                             src_iens))
                            enkf_node_copy(config_node, source_case,
                                           target_case, src_id, target_id);

                        target_state_map.set(src_iens, STATE_INITIALIZED);
                    }
                    src_iens++;
                }
            }
            enkf_fs_fsync(target_case);
        },
        py::arg("self"), py::arg("ensemble_config"), py::arg("target_case"),
        py::arg("report_step"), py::arg("node_list"), py::arg("iactive"));
}
