#include <stdlib.h>
#include <string.h>

#include <ert/util/string_util.h>
#include <ert/util/util.h>

#include <ert/ecl/ecl_endian_flip.h>
#include <ert/ecl/ecl_grid.h>

#include <ert/enkf/config_keys.hpp>
#include <ert/enkf/enkf_defaults.hpp>
#include <ert/enkf/enkf_types.hpp>
#include <ert/enkf/field_config.hpp>
#include <ert/enkf/field_trans.hpp>

/**
   About transformations and truncations
   -------------------------------------

   The values of the fields data can be automagically manipulated through two methods:

   * You can specify a min and a max value which will serve as truncation.

   * You can specify transformation functions which are applied to the field as follows:

     init_transform: This function is applied to the field when the
        field is loaded the first time, i.e. initialized. It is *NOT*
        applied under subsequent loads of dynamic fields during the
        execution.

     output_transform: This function is applied to the field before it
        is exported to eclipse.

     input_transform: This function is applied each time a field is
        loaded in from the forward model; i.e. this transformation
        applies to dynamic fields.



                                                            _______________________________         ___
                                                           /                               \        /|\
                                                           | Forward model (i.e. ECLIPSE)  |         |
                                                           | generates dynamic fields like |         |
                                                           | PRESSURE and SATURATIONS      |         |
                                                           \_______________________________/         |     This code is run
                                                                          |                          |     every time a field
                                                                          |                          |     is loaded FROM the
                                                                         \|/                         |     forward model into
                                                                          |                          |     EnKF.
                                                                  ________|_________                 |
                                                                 /                  \                |
                                                                 | Input transform  |                |
                                                                 \__________________/                |
                                                                          |                          |
                                                                          |                          |
                                                                         \|/                         |
                                                                          |                          |
                                                          ________________|__________________      _\|/_
_______________                       ___________        /                                   \
               \                     /           \       |  The internal representation      |
 Geo Modelling |                     | init-     |       |  of the field. This (should)      |
 creates a     |==>===============>==| transform |===>===|  be a normally distributed        |
 realization   |                     |           |       |  variable suitable for updates    |
_______________/                     \___________/       |  with EnKF.                       |
                                                         \___________________________________/   ___
|<----   This path is ONLY executed during INIT ------->|                  |                     /|\
         Observe that there is no truncation                              \|/                     |
         on load.                                                 _________|__________            |
                                                                 /                    \           |   This code is run
                                                                 |  Output transform  |           |   every time a field
                                                                 \____________________/           |   is exported from
                                                                           |                      |   enkf to the forward
                                                                          \|/                     |   model - i.e. ECLIPSE.
                                                                  _________|__________            |
                                                                 /                    \           |
                                                                 | Truncate min/max   |           |
                                                                 \____________________/           |
                                                                           |                      |
                                                                          \|/                     |
                                                                  _________|__________            |
                                                                 /                    \           |
                                                                 |    FORWARD MODEL   |           |
                                                                 \____________________/         _\|/_


Observe the following convention:

    global_index:  [0 , nx*ny*nz)
    active_index:  [0 , nactive)
*/
struct field_config_struct {
    /** Name/key ... */
    char *ecl_kw_name;
    /** The number of elements in the three directions. */
    int data_size, nx, ny, nz;
    /** Whether the data contains only active cells or active and inactive
     * cells */
    bool keep_inactive_cells;
    /** A shared reference to the grid this field is defined on. */
    ecl_grid_type *grid;
    bool private_grid;

    /** How the field should be trunacted before exporting for simulation, and
     * for the inital import. OR'd combination of truncation_type from enkf_types.h*/
    int truncation;
    /** The min value used in truncation. */
    double min_value;
    /** The maximum value used in truncation. */
    double max_value;

    field_file_format_type export_format;
    field_file_format_type import_format;
    char *output_field_name;
    /** See doc of functions field_config_set_key() / field_config_enkf_OFF() */
    bool __enkf_mode;
    bool write_compressed;

    field_type_enum type;
    /** Internalize a (pointer to) a table of the available transformation functions. */
    field_trans_table_type *trans_table;
    /** Function to apply to the data before they are exported - NULL: no transform. */
    field_func_type *output_transform;
    /** Function to apply on the data when they are loaded the first time -
     * i.e. initialized. NULL : no transform*/
    field_func_type *init_transform;
    /** Function to apply on the data when they are loaded from the forward
     * model - i.e. for dynamic data. */
    field_func_type *input_transform;

    char *output_transform_name;
    char *init_transform_name;
    char *input_transform_name;
};

/**
   This function takes a field_file_format_type variable, and returns
   a string containing a default extension for files of this type. For
   ecl_kw_file it will return NULL, i.e. no default extension.

     rms_roff_file   => ROFF
     ecl_grdecl_file => GRDECL
     ecl_kw_file_xxx => NULL

   It will return UPPERCASE or lowercase depending on the value of the
   second argument.
*/
const char *field_config_default_extension(field_file_format_type file_type,
                                           bool upper_case) {
    if (file_type == RMS_ROFF_FILE) {
        if (upper_case)
            return "ROFF";
        else
            return "roff";
    } else if (file_type == ECL_GRDECL_FILE) {
        if (upper_case)
            return "GRDECL";
        else
            return "grdecl";
    } else
        return NULL;
}

field_file_format_type
field_config_default_export_format(const char *filename) {
    field_file_format_type export_format = FILE_FORMAT_NULL;
    if (filename != NULL) {
        export_format =
            ECL_KW_FILE_ALL_CELLS; /* Suitable for PERMX/PORO/... ; when this export format is
                                                used IMPORT must be used in the datafile instead of
                                                INCLUDE. This gives faster ECLIPSE startup time, but is
                                                (unfortunately) quite unstandard. */

        char *extension;
        util_alloc_file_components(filename, NULL, NULL, &extension);
        if (extension != NULL) {
            util_strupr(extension);
            if (strcmp(extension, "GRDECL") == 0)
                export_format = ECL_GRDECL_FILE;
            else if (strcmp(extension, "ROFF") == 0)
                export_format = RMS_ROFF_FILE;

            free(extension);
        }
    }
    return export_format;
}

field_file_format_type
field_config_get_export_format(const field_config_type *field_config) {
    return field_config->export_format;
}

const char *
field_config_get_output_file_name(const field_config_type *field_config) {
    return field_config->output_field_name;
}

/**
   Will return the name of the init_transform function, or NULL if no
   init_transform function has been registered.
*/
const char *
field_config_get_init_transform_name(const field_config_type *field_config) {
    return field_config->init_transform_name;
}

const char *
field_config_get_output_transform_name(const field_config_type *field_config) {
    return field_config->output_transform_name;
}

/**
   IFF the @private_grid parameter is true, the field_config instance
   will take ownership of grid, i.e. freeing it in
   field_config_free().

   The field_config object exports a field_config_set_grid() function,
   but that is actually quite misleading. If this function is called
   during a run there are many other dependencies which must also be
   updated, which are not handled.
*/
void field_config_set_grid(field_config_type *config, ecl_grid_type *grid,
                           bool private_grid) {
    if ((config->private_grid) && (config->grid != NULL))
        ecl_grid_free(config->grid);

    config->grid = grid;
    config->private_grid = private_grid;

    ecl_grid_get_dims(grid, &config->nx, &config->ny, &config->nz, NULL);
    config->data_size = field_config_get_data_size_from_grid(config);
}

const char *field_config_get_grid_name(const field_config_type *config) {
    return ecl_grid_get_name(config->grid);
}

/*
  The return value from this function is hardly usable.
*/
field_config_type *field_config_alloc_empty(const char *ecl_kw_name,
                                            ecl_grid_type *ecl_grid,
                                            bool keep_inactive_cells) {

    field_config_type *config =
        (field_config_type *)util_malloc(sizeof *config);

    config->keep_inactive_cells = keep_inactive_cells;
    config->ecl_kw_name = util_alloc_string_copy(ecl_kw_name);
    config->output_field_name = NULL;
    config->private_grid = false;
    config->__enkf_mode = true;
    config->grid = NULL;
    config->write_compressed = true;
    config->type = UNKNOWN_FIELD_TYPE;

    config->output_transform = NULL;
    config->input_transform = NULL;
    config->init_transform = NULL;
    config->output_transform_name = NULL;
    config->input_transform_name = NULL;
    config->init_transform_name = NULL;

    config->truncation = TRUNCATE_NONE;
    config->trans_table = field_trans_table_alloc();

    field_config_set_grid(
        config, ecl_grid,
        false); /* The grid is (currently) set on allocation and can NOT be updated afterwards. */
    return config;
}

static void field_config_set_init_transform(field_config_type *config,
                                            const char *__init_transform_name) {
    const char *init_transform_name = NULL;
    if (field_trans_table_has_key(config->trans_table, __init_transform_name))
        init_transform_name = __init_transform_name;
    else if (__init_transform_name != NULL) {
        fprintf(stderr,
                "Sorry: the field transformation function:%s is not recognized "
                "\n\n",
                __init_transform_name);
        field_trans_table_fprintf(config->trans_table, stderr);
        util_exit("Exiting ... \n");
    }

    config->init_transform_name = util_realloc_string_copy(
        config->init_transform_name, init_transform_name);
    if (init_transform_name != NULL)
        config->init_transform =
            field_trans_table_lookup(config->trans_table, init_transform_name);
    else
        config->init_transform = NULL;
}

static void
field_config_set_output_transform(field_config_type *config,
                                  const char *__output_transform_name) {
    const char *output_transform_name = NULL;
    if (field_trans_table_has_key(config->trans_table, __output_transform_name))
        output_transform_name = __output_transform_name;
    else if (__output_transform_name) {
        fprintf(stderr,
                "Sorry: the field transformation function:%s is not recognized "
                "\n\n",
                __output_transform_name);
        field_trans_table_fprintf(config->trans_table, stderr);
        util_exit("Exiting ... \n");
    }

    config->output_transform_name = util_realloc_string_copy(
        config->output_transform_name, output_transform_name);
    if (output_transform_name != NULL)
        config->output_transform = field_trans_table_lookup(
            config->trans_table, output_transform_name);
    else
        config->output_transform = NULL;
}

void field_config_update_field(
    field_config_type *config, int truncation, double min_value,
    double max_value,
    field_file_format_type
        export_format, /* This can be guessed with the field_config_default_export_format( ecl_file ) function. */
    const char *init_transform, const char *input_transform,
    const char *output_transform, const char *output_field_name) {

    field_config_set_truncation(config, truncation, min_value, max_value);
    config->export_format = export_format;
    config->import_format =
        UNDEFINED_FORMAT; /* Guess from filename when loading. */
    config->type = ECLIPSE_PARAMETER;
    field_config_set_init_transform(config, init_transform);
    field_config_set_output_transform(config, output_transform);
    config->output_field_name = util_alloc_string_copy(output_field_name);
}

/**
   Requirements:

   ECLIPSE_PARAMETER: export_format != UNDEFINED_FORMAT

   ECLIPSE_RESTART  : Validation can be finalized at the enkf_config_node level.

   GENERAL          : export_format != UNDEFINED_FORMAT
*/
bool field_config_is_valid(const field_config_type *field_config) {
    bool valid = true;

    switch (field_config->type) {
    case ECLIPSE_PARAMETER:
        if (field_config->export_format == UNDEFINED_FORMAT)
            valid = false;
        break;
    case ECLIPSE_RESTART:
        break;
    case GENERAL:
        if (field_config->export_format == UNDEFINED_FORMAT)
            valid = false;
        break;
    default:
        util_abort("%s: Internal inconsistency in field config \n", __func__);
    }
    return valid;
}

field_type_enum field_config_get_type(const field_config_type *config) {
    return config->type;
}

/**
  Observe that the indices are zero-based, in contrast to those used
  by eclipse which are based on one.

  This function will return an index in the interval: [0...nactive),
  and -1 if i,j,k correspond to an inactive cell.
*/
int field_config_active_index(const field_config_type *config, int i, int j,
                              int k) {
    return ecl_grid_get_active_index3(config->grid, i, j, k);
}

int field_config_global_index(const field_config_type *config, int i, int j,
                              int k) {
    return ecl_grid_get_global_index3(config->grid, i, j, k);
}

/**
    This function checks that i,j,k are in the intervals [0..nx),
    [0..ny) and [0..nz). It does *NOT* check if the corresponding
    index is active.
*/
bool field_config_ijk_valid(const field_config_type *config, int i, int j,
                            int k) {
    return ecl_grid_ijk_valid(config->grid, i, j, k);
}

/**
    This function checks that i,j,k are in the intervals [0..nx),
    [0..ny) and [0..nz) AND that the corresponding cell is active. If
    the function returns false it is impossible to differentiate
    between (i,j,k) values which are out of bounds and an inactive
    cell.
*/
bool field_config_ijk_active(const field_config_type *config, int i, int j,
                             int k) {
    if (ecl_grid_ijk_valid(config->grid, i, j, k)) {
        int active_index = ecl_grid_get_active_index3(config->grid, i, j, k);

        if (active_index >= 0)
            return true;
        else
            return false;
    } else
        return false;
}

void field_config_set_truncation(field_config_type *config, int truncation,
                                 double min_value, double max_value) {
    config->truncation = truncation;
    config->min_value = min_value;
    config->max_value = max_value;
}

int field_config_get_truncation_mode(const field_config_type *config) {
    return config->truncation;
}

double field_config_get_truncation_min(const field_config_type *config) {
    return config->min_value;
}

double field_config_get_truncation_max(const field_config_type *config) {
    return config->max_value;
}

void field_config_free(field_config_type *config) {
    free(config->ecl_kw_name);
    free(config->input_transform_name);
    free(config->output_transform_name);
    free(config->init_transform_name);
    if ((config->private_grid) && (config->grid != NULL))
        ecl_grid_free(config->grid);
    field_trans_table_free(config->trans_table);
    free(config);
}

int field_config_get_volume(const field_config_type *config) {
    return config->nx * config->ny * config->nz;
}

int field_config_get_data_size_from_grid(const field_config_type *config) {
    return config->keep_inactive_cells ? ecl_grid_get_global_size(config->grid)
                                       : ecl_grid_get_active_size(config->grid);
}

int field_config_get_byte_size(const field_config_type *config) {
    int num_cells = field_config_get_data_size_from_grid(config);
    return num_cells * sizeof(float);
}

/**
   Returns true / false whether a cell is active.
*/
bool field_config_active_cell(const field_config_type *config, int i, int j,
                              int k) {
    int active_index = field_config_active_index(config, i, j, k);
    if (active_index >= 0)
        return true;
    else
        return false;
}

void field_config_get_dims(const field_config_type *config, int *nx, int *ny,
                           int *nz) {
    *nx = config->nx;
    *ny = config->ny;
    *nz = config->nz;
}

int field_config_get_nx(const field_config_type *config) { return config->nx; }

int field_config_get_ny(const field_config_type *config) { return config->ny; }

int field_config_get_nz(const field_config_type *config) { return config->nz; }

/**
   The field_config and field objects are mainly written for use in
   the enkf application. In that setting a field instance is *NOT*
   allowed to write on it's field_config object.

   However, when used in a stand-alone application, i.e. in the
   field_convert program, it is desirable for the field object to be
   allowed to write to / update the field_config object. In an attempt
   to make this reasonably safe you must first call
   field_config_enkf_OFF() to signal that you know what you are doing.

   After you have called field_config_enkf_OFF() you can subsequently
   call field_config_set_key() to change the key of the field_config
   object. This will typically be interesting when an unknown file is
   loaded.

   Currently only the roff loader supports set operations on the
   key. Also it is essential to observe that this will break **HARD**
   is the file contains several parameters
*/
void field_config_set_key(field_config_type *config, const char *key) {
    if (config->__enkf_mode)
        util_abort("%s: internal error - must call field_config_enkf_OFF() "
                   "prior to calling: %s()\n",
                   __func__, __func__);
    /*
    Should be locked to protect against concurrent access.
  */
    config->ecl_kw_name = util_realloc_string_copy(config->ecl_kw_name, key);
}

const char *field_config_get_key(const field_config_type *field_config) {
    return field_config->ecl_kw_name;
}

bool field_config_keep_inactive_cells(const field_config_type *config) {
    return config->keep_inactive_cells;
}

void field_config_enkf_OFF(field_config_type *config) {
    if (config->__enkf_mode)
        fprintf(stderr,
                "** Warning: turning off EnKF mode for field:%s - you better "
                "know what you are doing! **\n",
                config->ecl_kw_name);
    config->__enkf_mode = false;
}

bool field_config_enkf_mode(const field_config_type *config) {
    return config->__enkf_mode;
}

field_func_type *
field_config_get_output_transform(const field_config_type *config) {
    return config->output_transform;
}

field_func_type *
field_config_get_init_transform(const field_config_type *config) {
    return config->init_transform;
}

/**
   Asserts that two fields can be combined in a binary operation.
*/
void field_config_assert_binary(const field_config_type *config1,
                                const field_config_type *config2,
                                const char *caller) {
    const int size1 = config1->data_size;
    const int size2 = config2->data_size;

    if (size1 != size2)
        util_abort("%s: fields not equal enough - failure in:%s \n", __func__,
                   caller);
}

/*
   @brief Parses a string of the type "1,5,6", and returns the indices i,j,k
   by reference.

   @param[in] index_key The index key string, ie. "1,5,6"
   @param[out] i The i index as an int, ie. 1.
   @param[out] j The j index as an int, ie. 5.
   @param[out] k The k index as an int, ie. 6.
   @return True if parsing was successfull.
*/
bool field_config_parse_user_key__(const char *index_key, int *i, int *j,
                                   int *k) {
    int length;
    {
        int_vector_type *indices = string_util_alloc_value_list(index_key);
        length = int_vector_size(indices);

        if (length == 3) {
            *i = int_vector_iget(indices, 0) - 1;
            *j = int_vector_iget(indices, 1) - 1;
            *k = int_vector_iget(indices, 2) - 1;
        }

        int_vector_free(indices);
    }
    if (length == 3)
        return true;
    else
        return false;
}

/*
   @brief Parses a string of the type "1,5,6", and returns the indices i,j,k
   by reference.


   @param[in] config The field_config which is the context for whether the
       i,j,k values are valid.
   @param[in] index_key The index key string, ie. "1,5,6"
   @param[out] i The i index as an int, ie. 1.
   @param[out] j The j index as an int, ie. 5.
   @param[out] k The k index as an int, ie. 6.
   @return The return value of the function is
     whether the string constitutes a valid cell:
      0: All is OK.
      1: The string could not pe parsed to three valid integers.
      2: ijk are not in the grid.
      3: ijk correspond to an inactive cell.
      In cases 2 & 3 the i,j,k are valid (in the string-parsing sense).
      The input string is assumed to have offset one, and the return
      values (by reference) are offset zero.
*/
int field_config_parse_user_key(const field_config_type *config,
                                const char *index_key, int *i, int *j, int *k) {
    int return_value = 0;

    if (field_config_parse_user_key__(index_key, i, j, k)) {
        if (field_config_ijk_valid(config, *i, *j, *k)) {
            int active_index = field_config_active_index(config, *i, *j, *k);
            if (active_index < 0)
                return_value = 3; /* ijk corresponds to an inactive cell. */
        } else
            return_value = 2; /* ijk is outside the grid. */
    } else
        return_value = 1; /* Could not be parsed to three integers. */

    return return_value;
}

ecl_grid_type *field_config_get_grid(const field_config_type *config) {
    return config->grid;
}

CONFIG_GET_ECL_KW_NAME(field);
GET_DATA_SIZE(field)
VOID_GET_DATA_SIZE(field)
VOID_FREE(field_config)
