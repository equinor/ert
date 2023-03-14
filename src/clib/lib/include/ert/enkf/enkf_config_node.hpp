#ifndef ERT_ENKF_CONFIG_NODE_H
#define ERT_ENKF_CONFIG_NODE_H

#include <ert/util/hash.h>
#include <ert/util/stringlist.h>

#include <ert/config/config_parser.hpp>

#include <ert/ecl/ecl_grid.hpp>

#include <ert/enkf/enkf_macros.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/field_trans.hpp>
#include <ert/enkf/gen_data_config.hpp>
#include <ert/enkf/summary_config.hpp>
#include <ert/res_util/path_fmt.hpp>

typedef void(config_free_ftype)(void *);
typedef int(get_data_size_ftype)(const void *);
typedef void(config_fprintf_ftype)(const void *, enkf_var_type, FILE *);

typedef struct enkf_config_node_struct enkf_config_node_type;

struct enkf_config_node_struct {
    ert_impl_type impl_type;
    enkf_var_type var_type;
    /** Should the (parameter) node be initialized by loading results from the
     * Forward model? */
    bool forward_init;

    /** Keys of observations which observe this node. */
    stringlist_type *obs_keys;
    char *key;
    /** Format used to create files for initialization. */
    path_fmt_type *init_file_fmt;
    /** Format used to load in file from forward model - one %d (if present) is
     * replaced with report_step. */
    path_fmt_type *enkf_infile_fmt;
    /** Name of file which is written by EnKF, and read by the forward model. */
    path_fmt_type *enkf_outfile_fmt;
    /** This points to the config object of the actual implementation. */
    void *data;

    /** Function pointer to ask the underlying config object of the size - i.e.
     * number of elements. */
    get_data_size_ftype *get_data_size;
    config_free_ftype *freef;
};

enkf_config_node_type *
enkf_config_node_alloc_GEN_DATA_result(const char *key,
                                       gen_data_file_format_type input_format,
                                       const char *enkf_infile_fmt);

extern "C" enkf_config_node_type *
enkf_config_node_alloc(enkf_var_type, ert_impl_type, bool, const char *,
                       const char *, const char *, const char *, void *);

extern "C" enkf_config_node_type *
enkf_config_node_alloc_summary(const char *key, load_fail_type load_fail);

extern "C" void enkf_config_node_update_parameter_field(
    enkf_config_node_type *config_node, const char *enkf_outfile_fmt,
    const char *init_file_fmt, int truncation, double value_min,
    double value_max, const char *init_transform, const char *output_transform);

extern "C" enkf_config_node_type *
enkf_config_node_alloc_GEN_DATA_everest(const char *key,
                                        const char *result_file_fmt,
                                        const int_vector_type *report_steps);

extern "C" void enkf_config_node_update_general_field(
    enkf_config_node_type *config_node, const char *enkf_outfile_fmt,
    const char *enkf_infile_fmt, const char *init_file_fmt, int truncation,
    double value_min, double value_max, const char *init_transform,
    const char *input_transform, const char *output_transform);

extern "C" enkf_config_node_type *
enkf_config_node_alloc_field(const char *key, ecl_grid_type *ecl_grid,
                             bool forward_init);
int enkf_config_node_get_data_size(const enkf_config_node_type *node,
                                   int report_step);
char *enkf_config_node_alloc_infile(const enkf_config_node_type *, int);
extern "C" const stringlist_type *
enkf_config_node_get_obs_keys(const enkf_config_node_type *);
extern "C" void enkf_config_node_free(enkf_config_node_type *);
extern "C" ert_impl_type
enkf_config_node_get_impl_type(const enkf_config_node_type *);
extern "C" enkf_var_type
enkf_config_node_get_var_type(const enkf_config_node_type *);
extern "C" void *enkf_config_node_get_ref(const enkf_config_node_type *);
extern "C" const char *enkf_config_node_get_key(const enkf_config_node_type *);
extern "C" const char *
enkf_config_node_get_enkf_outfile(const enkf_config_node_type *conifg_node);
extern "C" const char *
enkf_config_node_get_enkf_infile(const enkf_config_node_type *config_node);
extern "C" const char *
enkf_config_node_get_init_file_fmt(const enkf_config_node_type *config_node);
char *enkf_config_node_alloc_initfile(const enkf_config_node_type *node,
                                      const char *path, int iens);
/*
    The enkf_node_free() function declaration is in the enkf_config_node.h header,
    because the enkf_config_node needs to know how to free the min_std node.
    That's not really the reason, because min_std doesn't exist anymore, but
    the function declaration is needed anyways.
  */

extern "C" bool
enkf_config_node_use_forward_init(const enkf_config_node_type *config_node);

extern "C" PY_USED enkf_config_node_type *
enkf_config_node_alloc_GEN_DATA_full(const char *node_key,
                                     const char *result_file,
                                     gen_data_file_format_type input_format,
                                     const int_vector_type *report_steps);

extern "C" PY_USED enkf_config_node_type *enkf_config_node_alloc_GEN_KW_full(
    const char *node_key, bool forward_init, const char *gen_kw_format,
    const char *template_file, const char *enkf_outfile,
    const char *parameter_file, const char *init_file_fmt);

extern "C" PY_USED enkf_config_node_type *enkf_config_node_alloc_SURFACE_full(
    const char *node_key, bool forward_init, const char *output_file,
    const char *base_surface, const char *init_file_fmt);

VOID_FREE_HEADER(enkf_config_node);
#endif
