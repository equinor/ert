#include <stdlib.h>

#include <ert/util/buffer.h>
#include <ert/util/util.h>
#include <ert/util/vector.h>

#include <ert/enkf/enkf_node.hpp>
#include <ert/enkf/ext_param.hpp>
#include <ert/enkf/field.hpp>
#include <ert/enkf/gen_kw.hpp>
#include <ert/enkf/surface.hpp>
#include <ert/python.hpp>

/**
   A small illustration (says more than thousand words ...) of how the
   enkf_node, enkf_config_node, field[1] and field_config[1] objects
   are linked.


   ================
   |              |   o-----------
   |  ================           |                =====================
   |  |              |   o--------                |                   |
   |  |  ================        |------------->  |                   |
   |  |  |              |        |                |  enkf_config_node |
   |  |  |              |        |                |                   |
   ===|  |  enkf_node   |  o------                |                   |
   o  |  |              |                         |                   |
   |  ===|              |                         =====================
   |   o |              |                                  o
   |   | ================                                  |
   |   |       o                                           |
   |   \       |                                           |
   |    \      |                                           |
   |    |      |                                           |
   |    |      |                                           |
   |    |      |                                           |
   |    |      |                                           |
  \|/   |      |                                           |
  ======|======|==                                        \|/
  |    \|/     | |   o-----------
  |  ==========|=====           |                =====================
  |  |        \|/   |   o--------                |                   |
  |  |  ================        |------------->  |                   |
  |  |  |              |        |                |  field_config     |
  |  |  |              |        |                |                   |
  ===|  |  field       |  o------                |                   |
  |     |              |                         |                   |
  ===   |              |                         =====================
        |              |
        ================


   To summarize in words:

   * The enkf_node object is an abstract object, which again contains
   a spesific enkf_object, like e.g. the field objects shown
   here. In general we have an ensemble of enkf_node objects.

   * The enkf_node objects contain a pointer to an enkf_config_node
   object.

   * The enkf_config_node object contains a pointer to the specific
   config object, i.e. field_config in this case.

   * All the field objects contain a pointer to a field_config object.


   [1]: field is just an example, and could be replaced with any of
   the enkf object types.

   A note on memory
   ================

   The enkf_nodes can consume large amounts of memory, and for large
   models/ensembles we have a situation where not all the
   members/fields can be in memory simultaneously - such low-memory
   situations are not really supported at the moment, but we have
   implemented some support for such problems:

   o All enkf objects should have a xxx_realloc_data() function. This
   function should be implemented in such a way that it is always
   safe to call, i.e. if the object already has allocated data the
   function should just return.

   o All enkf objects should implement a xxx_free_data()
   function. This function free the data of the object, and set the
   data pointer to NULL.


   The following 'rules' apply to the memory treatment:
   ----------------------------------------------------

   o Functions writing to memory can always be called, and it is their
   responsibility to allocate memory before actually writing on it. The
   writer functions are:

   enkf_node_initialize()
   enkf_node_forward_load()

   The (re)allocation of data is done at
   the enkf_node level, and **NOT** in the low level object
   (altough that is where it is eventually done of course).

   o When it comes to functions reading memory it is a bit more
   tricky. It could be that if the functions are called without
   memory, that just means that the object is not active or
   something (and the function should just return). On the other
   hand trying to read a NULL pointer does indicate that program
   logic is not fully up to it? And should therefore maybe be
   punished?

*/
struct enkf_node_struct {
    alloc_ftype *alloc;
    has_data_ftype *has_data;

    serialize_ftype *serialize;
    deserialize_ftype *deserialize;
    read_from_buffer_ftype *read_from_buffer;
    write_to_buffer_ftype *write_to_buffer;
    initialize_ftype *initialize;
    node_free_ftype *freef;
    node_copy_ftype *copy;

    bool vector_storage;

    /** The (hash)key this node is identified with. */
    char *node_key;

    /** A pointer to the underlying enkf_object, i.e. gen_kw_type instance, or
     * a field_type instance or ... */
    void *data;
    /** A pointer to a enkf_config_node instance (which again cointans a
     * pointer to the config object of data). */
    const enkf_config_node_type *config;
};

const enkf_config_node_type *enkf_node_get_config(const enkf_node_type *node) {
    return node->config;
}

/*
  All the function pointers REALLY should be in the config object ...
*/

#define FUNC_ASSERT(func)                                                      \
    if (func == NULL)                                                          \
        util_abort("%s: function handler: %s not registered for node:%s - "    \
                   "aborting\n",                                               \
                   __func__, #func, enkf_node->node_key);

void enkf_node_alloc_domain_object(enkf_node_type *enkf_node) {
    FUNC_ASSERT(enkf_node->alloc);
    {
        if (enkf_node->data != NULL)
            enkf_node->freef(enkf_node->data);
        enkf_node->data = enkf_node->alloc(
            enkf_config_node_get_ref(enkf_node->config)); // CXX_CAST_ERROR
    }
}

ert_impl_type enkf_node_get_impl_type(const enkf_node_type *enkf_node) {
    return enkf_config_node_get_impl_type(enkf_node->config);
}

void *enkf_node_value_ptr(const enkf_node_type *enkf_node) {
    return enkf_node->data;
}

bool enkf_node_forward_init(enkf_node_type *enkf_node,
                            const std::string &run_path, int iens) {
    char *init_file = enkf_config_node_alloc_initfile(enkf_node->config,
                                                      run_path.c_str(), iens);
    bool init = enkf_node->initialize(enkf_node->data, iens, init_file);
    free(init_file);
    return init;
}

static bool enkf_node_store_buffer(enkf_node_type *enkf_node, enkf_fs_type *fs,
                                   int report_step, int iens) {
    FUNC_ASSERT(enkf_node->write_to_buffer);
    {
        bool data_written;
        buffer_type *buffer = buffer_alloc(100);
        const enkf_config_node_type *config_node =
            enkf_node_get_config(enkf_node);
        buffer_fwrite_time_t(buffer, time(NULL));
        data_written =
            enkf_node->write_to_buffer(enkf_node->data, buffer, report_step);
        if (data_written) {
            const char *node_key = enkf_config_node_get_key(config_node);
            enkf_var_type var_type = enkf_config_node_get_var_type(config_node);

            if (enkf_node->vector_storage)
                enkf_fs_fwrite_vector(fs, buffer, node_key, var_type, iens);
            else
                enkf_fs_fwrite_node(fs, buffer, node_key, var_type, report_step,
                                    iens);
        }
        buffer_free(buffer);
        return data_written;
    }
}

bool enkf_node_store_vector(enkf_node_type *enkf_node, enkf_fs_type *fs,
                            int iens) {
    return enkf_node_store_buffer(enkf_node, fs, -1, iens);
}

bool enkf_node_store(enkf_node_type *enkf_node, enkf_fs_type *fs,
                     node_id_type node_id) {
    if (enkf_node->vector_storage)
        return enkf_node_store_vector(enkf_node, fs, node_id.iens);
    else
        return enkf_node_store_buffer(enkf_node, fs, node_id.report_step,
                                      node_id.iens);
}

bool enkf_node_try_load(enkf_node_type *enkf_node, enkf_fs_type *fs,
                        node_id_type node_id) {
    if (enkf_node_has_data(enkf_node, fs, node_id)) {
        enkf_node_load(enkf_node, fs, node_id);
        return true;
    } else
        return false;
}

static void enkf_node_buffer_load(enkf_node_type *enkf_node, enkf_fs_type *fs,
                                  int report_step, int iens) {
    FUNC_ASSERT(enkf_node->read_from_buffer);
    {
        buffer_type *buffer = buffer_alloc(100);
        const enkf_config_node_type *config_node =
            enkf_node_get_config(enkf_node);
        const char *node_key = enkf_config_node_get_key(config_node);
        enkf_var_type var_type = enkf_config_node_get_var_type(config_node);

        if (enkf_node->vector_storage)
            enkf_fs_fread_vector(fs, buffer, node_key, var_type, iens);
        else
            enkf_fs_fread_node(fs, buffer, node_key, var_type, report_step,
                               iens);

        buffer_fskip_time_t(buffer);

        enkf_node->read_from_buffer(enkf_node->data, buffer, fs, report_step);
        buffer_free(buffer);
    }
}

void enkf_node_load_vector(enkf_node_type *enkf_node, enkf_fs_type *fs,
                           int iens) {
    enkf_node_buffer_load(enkf_node, fs, -1, iens);
}

void enkf_node_load(enkf_node_type *enkf_node, enkf_fs_type *fs,
                    node_id_type node_id) {
    if (enkf_node->vector_storage)
        enkf_node_load_vector(enkf_node, fs, node_id.iens);
    else
        /* Normal load path */
        enkf_node_buffer_load(enkf_node, fs, node_id.report_step, node_id.iens);
}

bool enkf_node_try_load_vector(enkf_node_type *enkf_node, enkf_fs_type *fs,
                               int iens) {
    if (enkf_fs_has_vector(fs, enkf_node->config->key,
                           enkf_node->config->var_type, iens)) {
        enkf_node_load_vector(enkf_node, fs, iens);
        return true;
    } else
        return false;
}

/**
  In the case of nodes with vector storage this function
  will load the entire vector.
*/
enkf_node_type *enkf_node_load_alloc(const enkf_config_node_type *config_node,
                                     enkf_fs_type *fs, node_id_type node_id) {
    if (enkf_config_node_get_impl_type(config_node) == SUMMARY) {
        if (enkf_fs_has_vector(fs, config_node->key, config_node->var_type,
                               node_id.iens)) {
            enkf_node_type *node = enkf_node_alloc(config_node);
            enkf_node_load(node, fs, node_id);
            return node;
        } else {
            util_abort("%s: could not load vector:%s from iens:%d\n", __func__,
                       enkf_config_node_get_key(config_node), node_id.iens);
            return NULL;
        }
    } else {
        if (enkf_fs_has_node(fs, config_node->key, config_node->var_type,
                             node_id.report_step, node_id.iens)) {
            enkf_node_type *node = enkf_node_alloc(config_node);
            enkf_node_load(node, fs, node_id);
            return node;
        } else {
            util_abort("%s: Could not load node: key:%s  iens:%d  report:%d \n",
                       __func__, enkf_config_node_get_key(config_node),
                       node_id.iens, node_id.report_step);
            return NULL;
        }
    }
}

void enkf_node_copy(const enkf_config_node_type *config_node,
                    enkf_fs_type *src_case, enkf_fs_type *target_case,
                    node_id_type src_id, node_id_type target_id) {

    enkf_node_type *enkf_node =
        enkf_node_load_alloc(config_node, src_case, src_id);

    enkf_node_store(enkf_node, target_case, target_id);
    enkf_node_free(enkf_node);
}

bool enkf_node_has_data(enkf_node_type *enkf_node, enkf_fs_type *fs,
                        node_id_type node_id) {
    int report_step = node_id.report_step;
    int iens = node_id.iens;
    const enkf_config_node_type *config = enkf_node->config;

    if (enkf_node->vector_storage) {
        FUNC_ASSERT(enkf_node->has_data);
        {
            // Try to load the vector.
            if (enkf_fs_has_vector(fs, config->key, config->var_type, iens)) {
                enkf_node_load_vector(enkf_node, fs, iens);

                // The vector is loaded. Check if we have the report_step/state asked for:
                return enkf_node->has_data(enkf_node->data, report_step);
            } else
                return false;
        }
    } else
        return enkf_fs_has_node(fs, config->key, config->var_type, report_step,
                                iens);
}

void enkf_node_serialize(enkf_node_type *enkf_node, enkf_fs_type *fs,
                         node_id_type node_id, const ActiveList *active_list,
                         Eigen::MatrixXd &A, int row_offset, int column) {

    FUNC_ASSERT(enkf_node->serialize);
    enkf_node_load(enkf_node, fs, node_id);
    enkf_node->serialize(enkf_node->data, node_id, active_list, A, row_offset,
                         column);
}

void enkf_node_deserialize(enkf_node_type *enkf_node, enkf_fs_type *fs,
                           node_id_type node_id, const ActiveList *active_list,
                           const Eigen::MatrixXd &A, int row_offset,
                           int column) {

    FUNC_ASSERT(enkf_node->deserialize);
    enkf_node->deserialize(enkf_node->data, node_id, active_list, A, row_offset,
                           column);
    enkf_node_store(enkf_node, fs, node_id);
}

/**
   The return value is whether any initialization has actually taken
   place. If the function returns false it is for instance not
   necessary to internalize anything.
*/
bool enkf_node_initialize(enkf_node_type *enkf_node, int iens) {
    if (enkf_node->initialize != NULL) {
        char *init_file =
            enkf_config_node_alloc_initfile(enkf_node->config, NULL, iens);
        bool init = enkf_node->initialize(enkf_node->data, iens, init_file);
        free(init_file);
        return init;
    } else
        return false; /* No init performed */
}

extern "C" void enkf_node_free(enkf_node_type *enkf_node) {
    if (enkf_node->freef != NULL)
        enkf_node->freef(enkf_node->data);
    free(enkf_node->node_key);
    free(enkf_node);
}

const char *enkf_node_get_key(const enkf_node_type *enkf_node) {
    return enkf_node->node_key;
}

#undef FUNC_ASSERT

static enkf_node_type *
enkf_node_alloc_empty(const enkf_config_node_type *config) {
    const char *node_key = enkf_config_node_get_key(config);
    ert_impl_type impl_type = enkf_config_node_get_impl_type(config);
    enkf_node_type *node = (enkf_node_type *)util_malloc(sizeof *node);
    node->vector_storage = (impl_type == SUMMARY);
    node->config = config;
    node->node_key = util_alloc_string_copy(node_key);
    node->data = NULL;

    node->alloc = NULL;
    node->initialize = NULL;
    node->freef = NULL;
    node->read_from_buffer = NULL;
    node->write_to_buffer = NULL;
    node->serialize = NULL;
    node->deserialize = NULL;
    node->has_data = NULL;

    switch (impl_type) {
    case (GEN_KW):
        node->alloc = gen_kw_alloc__;
        node->freef = gen_kw_free__;
        node->write_to_buffer = gen_kw_write_to_buffer__;
        node->read_from_buffer = gen_kw_read_from_buffer__;
        node->serialize = gen_kw_serialize__;
        node->deserialize = gen_kw_deserialize__;
        break;
    case (SURFACE):
        node->initialize = surface_initialize__;
        node->alloc = surface_alloc__;
        node->freef = surface_free__;
        node->read_from_buffer = surface_read_from_buffer__;
        node->write_to_buffer = surface_write_to_buffer__;
        node->serialize = surface_serialize__;
        node->deserialize = surface_deserialize__;
        break;
    case (FIELD):
        node->alloc = field_alloc__;
        node->initialize = field_initialize__;
        node->freef = field_free__;
        node->read_from_buffer = field_read_from_buffer__;
        node->write_to_buffer = field_write_to_buffer__;
        node->serialize = field_serialize__;
        node->deserialize = field_deserialize__;
        break;
    /* EXT_PARAM is used by Everest */
    case (EXT_PARAM):
        node->alloc = ext_param_alloc__;
        node->freef = ext_param_free__;
        node->write_to_buffer = ext_param_write_to_buffer__;
        node->read_from_buffer = ext_param_read_from_buffer__;
        break;
    default:
        util_abort("%s: implementation type: %d unknown - all hell is loose - "
                   "aborting \n",
                   __func__, impl_type);
    }
    return node;
}

enkf_node_type *enkf_node_alloc(const enkf_config_node_type *config) {
    enkf_node_type *node = enkf_node_alloc_empty(config);
    enkf_node_alloc_domain_object(node);
    return node;
}

enkf_node_type *enkf_node_deep_alloc(const enkf_config_node_type *config) {
    return enkf_node_alloc(config);
}

ERT_CLIB_SUBMODULE("enkf_node", m) {
    using namespace py::literals;

    m.def("try_load", [](Cwrap<enkf_node_type> self, Cwrap<enkf_fs_type> fs,
                         int report_step, int iens) {
        return enkf_node_try_load(self, fs, {report_step, iens});
    });

    m.def(
        "forward_init",
        [](Cwrap<enkf_node_type> enkf_node, const std::string &run_path,
           int iens) {
            return enkf_node_forward_init(enkf_node, run_path, iens);
        },
        "self"_a, "run_path"_a, "iens"_a);
    m.def("try_load", [](Cwrap<enkf_node_type> enkf_node,
                         Cwrap<enkf_fs_type> sim_fs, int report_step,
                         int iens) {
        return enkf_node_try_load(enkf_node, sim_fs,
                                  {.report_step = report_step, .iens = iens});
    });
    m.def("store", [](Cwrap<enkf_node_type> enkf_node,
                      Cwrap<enkf_fs_type> sim_fs, int report_step, int iens) {
        return enkf_node_store(enkf_node, sim_fs,
                               {.report_step = report_step, .iens = iens});
    });
    m.def("has_data", [](Cwrap<enkf_node_type> enkf_node,
                         Cwrap<enkf_fs_type> sim_fs, int report_step,
                         int iens) {
        return enkf_node_has_data(enkf_node, sim_fs,
                                  {.report_step = report_step, .iens = iens});
    });
}
