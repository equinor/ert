#ifndef ERT_ENKF_TYPES_H
#define ERT_ENKF_TYPES_H

#include <ert/tooling.hpp>

/**
  This enum signals the three different states a "cell" in
  observation/data node can be in:

  ACTIVE: The cell is active and should be used/updated in EnKF
    analysis.

  LOCAL_INACTIVE: The cell is not included in the current local
    analysis update_step

  DEACTIVATED: The cell has been deactivated by the functionality
    deactivating outliers.

*/
typedef enum {
    ACTIVE = 1,
    /** Not active in current local update scheme. */
    LOCAL_INACTIVE = 2,
    /** Deactivaed due to to small overlap, or... */
    DEACTIVATED = 3,
    MISSING = 4
} active_type; /* Set as missing by the forward model. */

/**
  The enkf_var_type enum defines logical groups of variables. All
  variables in the same group, i.e. 'parameter' are typically treated
  in the same manner. So the reason for creating this type is to be
  able to say for instance: "Load all dynamic_state variables".

  Observe that these are used as bitmask's, i.e. the numerical values
  must be a power of 2 series.
*/
typedef enum {
    INVALID_VAR = 0,
    /** A parameter which is updated with enkf: PORO, MULTFLT, ..*/
    PARAMETER = 1,
    /** Dynamic results which are NOT needed for a restart - i.e. well rates. */
    DYNAMIC_RESULT = 4,
    /** Index data - enum value is used for storage classification */
    INDEX_STATE = 16,
    EXT_PARAMETER = 32
} /* Parameter fully managed by external scope. */
enkf_var_type;

/*
   ert_impl_type are the actual node implementation types. Observe
   that one ert_impl_type can be used in several ways as
   enkf_var_type. For instance the pressure is implemented with a
   field, and behaves as a dynamic_state variable, on the other hand
   the permeability is also implemented as a field, but this is a
   parameter.

   These correspond to implementation types. The numbers are on disk,
   and should **NOT BE UPDATED**. The ERT_MIN_TYPE and MAX_TYPE
   identifiers are needed for the block_fs_driver.
*/

typedef enum {
    FIELD = 104,
    GEN_KW = 107,
    SUMMARY = 110,
    GEN_DATA = 113,
    SURFACE = 114,
    EXT_PARAM = 116
} ert_impl_type;

/*
   Should update the functions enkf_types_get_impl_name() and
   enkf_types_get_impl_type__() when this enum is updated.
   In addition to enkf_config_add_type().
*/

typedef enum {
    LOAD_SUCCESSFUL = 0,
    LOAD_FAILURE = 2,
    TIME_MAP_FAILURE = 3
} fw_load_status;

/**
   This enum is used to differentiate between different types of
   run. The point is that depending on this mode we can be more or
   less restrictive on the amount of input we require from the user.

   In mode enkf_assimlation ( which is the default ), we require quite
   a lot of info, whereas in the case screening_experiment we require
   less.

   screening_experiment:
      - SIZE
      - RUNPATH
      - ECLBASE
      - SCHEDULE_FILE
      - DATA_FILE
      - FORWARD_MODEL.

   ensemble_experiment:
      - ENSPATH
      - INIT_FILE (or estimation of EQUIL)

*/

typedef enum {
    JOB_NOT_STARTED = 0,
    JOB_RUN_FAILURE = 2,
    JOB_LOAD_FAILURE = 3,
} run_status_type;

typedef struct {
    int report_step;
    int iens;
} node_id_type;

typedef enum {
    STATE_UNDEFINED = 1,
    STATE_INITIALIZED = 2,
    STATE_HAS_DATA = 4,
    STATE_LOAD_FAILURE = 8,
    STATE_PARENT_FAILURE = 16
} realisation_state_enum;

typedef struct enkf_obs_struct enkf_obs_type;

const char *enkf_types_get_impl_name(ert_impl_type);

#endif
