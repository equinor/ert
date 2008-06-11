#ifndef __ENKF_TYPES_H__
#define __ENKF_TYPES_H__
#include <void_arg.h>
typedef double (transform_ftype) (double , const void_arg_type *);



/*
  Observe that seemingly random numbers are used in these enum definitions, 
  that is to be able to catch it if a wrong constant is used.
*/

typedef double enkf_float_type;

/*
  Observe that these are used as bitmask's, i.e. they must be power of 2 series.
*/
typedef enum {constant         = 1  , /* A parameter which is constant both in time, and between members                         */
	      static_parameter = 2  , /* A parameter which is not updated with enkf - can be different between different members */
              parameter        = 4  , /* A parameter which is updated with enkf                                                  */
	      ecl_restart      = 8  , /* Dynamic data - read from Eclipse restart files, typically PRESSURE and SATURATIONS      */
	      ecl_summary      = 16 , /* Dynamic data - summary data from Eclipse summary files                                  */ 
	      ecl_static       = 32 , /* Keywords like XCON++ from eclipse restart files - which are just dragged along          */ 
	      all_types        = 63 }  enkf_var_type;


/*

The static can be confusing - because the impl_type STATIC and the
enkf_var_type ecl_static symbolize perfectly the same thing. */

/* 
   For instance the pressure is implemented with a field, and behaves as a ecl_restart variable. The
   permeability is also implemented as a field, but this is a parameter.
*/

/*
  These correspond to implementation types.
*/
typedef enum {INVALID = 0   , 
	      STATIC = 100  ,
	      MULTZ   ,       /* 101 */
	      MULTFLT ,       /* 102 */
	      EQUIL   ,       /* 103 */   
	      FIELD   ,       /* 104 */ 
	      WELL    ,       /* 105 */ 
	      PGBOX   ,       /* 106 */ 
	      GEN_KW  ,       /* 107 */  
	      RELPERM ,       /* 108 */ 
	      HAVANA_FAULT,   /* 109 */ 
	      SUMMARY ,       /* 110 */
        TPGZONE         /* 111 */ }   enkf_impl_type;
/* 
   Should update the functions enkf_types_get_impl_name() and enkf_types_get_impl_type__() when this enum is updated.
*/



/*
  These types are logical types, describing how the parameter behaves in the EnKF
  loop.
*/


typedef enum {active_off = 200 , active_on , active_at , active_after , active_before} enkf_active_type;
typedef enum {abs_error = 0 , rel_error = 1 , rel_min_abs_error = 2} enkf_obs_error_type; /* Should not have enkf_ prefix */


/*
  typedef enum {WELL_OBS = 0 , POINT_OBS = 1} enkf_obs_type;
*/

typedef enum {nolog = 0 , log_input_mask = 1 , log_enkf_mask = 2 , log_output_mask = 4 , log_all = 7} enkf_logmode_enum;


typedef enum {store_none     = 0,
	      store_summary  = 1,
	      store_restart  = 2, 
	      store_data     = 4} ecl_store_enum;


enkf_impl_type    enkf_types_get_impl_type(const char * );
const char      * enkf_types_get_impl_name(enkf_impl_type );
enkf_impl_type    enkf_types_check_impl_type(const char * );
#endif
