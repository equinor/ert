#ifndef __ENKF_TYPES_H__
#define __ENKF_TYPES_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <arg_pack.h>




/*
  This enum signals the three different states a "cell" in
  observation/data node can be in:

  ACTIVE: The cell is active ans should be used/updated in EnKF
    analysis.

  LOCAL_INACTIVE: The cell is not included in the current local
    analysis ministep

  DEACTIVATED: The cell has been deactivated by the functionality
    deactivating outliers.

*/

typedef enum { ACTIVE         = 1,
               LOCAL_INACTIVE = 2,
               DEACTIVATED    = 3 } active_type;


/*
  The enkf_var_type enum defines logical groups of variables. All
  variables in the same group, i.e. 'parameter' are typically treated
  in the same manner. So the reason for creating this type is to be
  able to say for instance: "Load all dynamic_state variables".

  Observe that these are used as bitmask's, i.e. the numerical values
  must be a power of 2 series.
*/

typedef enum {INVALID_VAR      =  0  , /**/
              PARAMETER        =  1  , /* A parameter which is updated with enkf: PORO , MULTFLT , ..*/
	      DYNAMIC_STATE    =  2  , /* Dynamic data which are needed for a restart - i.e. pressure and saturations.  */
	      DYNAMIC_RESULT   =  4  , /* Dynamic results which are NOT needed for a restart - i.e. well rates. */
	      STATIC_STATE     =  8 }  /* Keywords like XCON++ from eclipse restart files - which are just dragged along          */ 
enkf_var_type; 
  
  

typedef enum { DEFAULT_KEEP    = 0,    /* Remove for enkf assimilation - keep for ensemble experiments. */
	       EXPLICIT_DELETE = 1,    /* Remove unconditionally */
	       EXPLICIT_KEEP   = 2}    /* keep unconditionally */
keep_runpath_type;






/* 
   enkf_impl_type are the actual node implementation types. Observe
   that one enkf_impl_type can be used in several ways as
   enkf_var_type. For instance the pressure is implemented with a
   field, and behaves as a dynamic_state variable, on the other hand
   the permeability is also implemented as a field, but this is a
   parameter.

   These correspond to implementation types. The numbers are on disk,
   and should **NOT BE UPDATED**. The __MIN_TYPE and __MAX_TYPE
   identifiers are needed for the block_fs_driver.
*/


  
/** HAVANA_FAULT = 109 - has been removed. */
typedef enum {INVALID 	       = 0   , 
              IMPL_TYPE_OFFSET = 100,
	      STATIC  	       = 100 ,       /* MULTZ has been removed & MULTFLT */ 
	      FIELD   	       = 104 ,       /* WELL has been removed  */
	      GEN_KW  	       = 107 ,       /* RELPERM has been removed & HAVANA_FAULT */
	      SUMMARY          = 110 ,       /* TPGZONE has been removed */
              GEN_DATA         = 113 ,       /* PILOT_POINT has been removed */
              MAX_IMPL_TYPE    = 113 } enkf_impl_type;

  

/* 
   Should update the functions enkf_types_get_impl_name() and
   enkf_types_get_impl_type__() when this enum is updated.
   In addition to enkf_config_add_type().
*/


typedef enum   {UNDEFINED   = 0 , 
		SERIALIZED  = 1,
		FORECAST    = 2, 
		ANALYZED    = 4,
		BOTH        = 6} state_enum;  /* It is important that both == (forecast + analyzed) */
  /**
     The state == both is used for output purposes (getting both forecast and analyzed).
  */




  /** 
      These are 2^n bitmasks.
  */

typedef enum { TRUNCATE_NONE   = 0,
	       TRUNCATE_MIN    = 1,
	       TRUNCATE_MAX    = 2 } truncation_type;




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

   enkf_assmilation:
      - RESULT_PATH

*/

typedef enum { ENKF_ASSIMILATION       = 1, 
	       ENSEMBLE_EXPERIMENT     = 2,
               ENSEMBLE_PREDICTION     = 3,
               INIT_ONLY               = 4} run_mode_type;
	       

 


/**
   This enum enumerates the different types of inflation which should
   be used. Observe that the actual variable used is not en enum
   instance, but rather an ordinary integer which is in general a sum
   of of the values listed in this enum.
*/


typedef enum { NO_INFLATION     = 0,
               SCALAR_INFLATION = 1,
               LOCAL_INFLATION  = 2} inflation_mode_type;



/*****************************************************************/
/*
  integer, intent(in) :: mode 
  ! first integer means (EnKF=1, SQRT=2)
  ! Second integer is pseudo inversion
  !  1=eigen value pseudo inversion of SS'+(N-1)R
  !  2=SVD subspace pseudo inversion of SS'+(N-1)R
  !  3=SVD subspace pseudo inversion of SS'+EE'
*/


typedef enum {
  ENKF_STANDARD    = 10,
  ENKF_SQRT        = 20,
  ENKF_KALMAN_GAIN = 30   /* No support for this yet ... */
} enkf_mode_type ;
	       

typedef enum {
  EIGEN_SS_N1_R = 1,
  SVD_SS_N1_R   = 2,
  SVD_SS_EE     = 3
} pseudo_inversion_type ;

/*****************************************************************/

/**
   This enum is used when we are setting up the dependencies between
   observations and variables. The modes all_active and inactive are
   sufficient information, for the values partly active we need
   additional information.

   The same type is used both for variables (PRESSURE/PORO/MULTZ/...)
   and for observations.
*/

typedef enum {
  ALL_ACTIVE 	= 1,       /* The variable/observation is fully active, i.e. all cells/all faults/all .. */
  INACTIVE   	= 2,       /* Fully inactive */
  PARTLY_ACTIVE = 3        /* Partly active - must supply additonal type spesific information on what is active.*/
} active_mode_type; 




/*****************************************************************/

typedef struct enkf_obs_struct enkf_obs_type;
  


const char      * enkf_types_get_var_name(enkf_var_type var_type);
enkf_impl_type    enkf_types_get_impl_type(const char * );
const char      * enkf_types_get_impl_name(enkf_impl_type );
enkf_impl_type    enkf_types_check_impl_type(const char * );

#ifdef __cplusplus
}
#endif
#endif
