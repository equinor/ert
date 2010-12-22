#ifndef __ENKF_MACROS_H__
#define __ENKF_MACROS_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <enkf_types.h>
#include <ecl_file.h>
#include <ecl_sum.h>
#include <enkf_serialize.h>
#include <active_list.h>
#include <matrix.h>
#include <log.h>
#include <meas_data.h>
#include <mzran.h>

#define CONFIG_STD_FIELDS \
int __type_id;            \
int data_size;



/*****************************************************************/

#define IS_INSTANCE(prefix,ID) \
bool prefix ## _is_instance__(const void * __arg) {                        \
  prefix ## _type * arg = (prefix ## _type *) __arg;               \
  if (arg->__type_id != ID)                                        \
     return false;                                                 \
  else                                                             \
     return true;                                                  \
}

#define IS_INSTANCE_HEADER(prefix)  bool prefix ## _is_instance__(const void * );

/******************************************************************/



/*****************************************************************/

#define VOID_CONFIG_FREE(prefix)            void prefix ## _config_free__(void *void_arg) { prefix ## _config_free((prefix ## _config_type *) void_arg); }
#define VOID_CONFIG_FREE_HEADER(prefix)     void prefix ## _config_free__(void *);

/*****************************************************************/

#define GET_DATA_SIZE(prefix)               int prefix ## _config_get_data_size (const prefix ## _config_type *arg) { return arg->data_size; }
#define GET_DATA_SIZE_HEADER(prefix)        int prefix ## _config_get_data_size (const prefix ## _config_type *arg);

#define VOID_GET_DATA_SIZE(prefix)               int prefix ## _config_get_data_size__ (const void * arg) {\
   prefix ## _config_type * config = prefix ## _config_safe_cast_const( arg ); \
   return prefix ## _config_get_data_size( config );                     \
}
#define VOID_GET_DATA_SIZE_HEADER(prefix)        int prefix ## _config_get_data_size__ (const void * arg);


/*****************************************************************/


#define VOID_ALLOC(prefix)                                                            \
void * prefix ## _alloc__(const void *void_config) {                                  \
  const prefix ## _config_type * config = prefix ## _config_safe_cast_const( void_config ); \
  return prefix ## _alloc(config);                                                    \
}

#define VOID_ALLOC_HEADER(prefix) void * prefix ## _alloc__(const void *);

/*****************************************************************/

#define VOID_STORE(prefix)                                        \
bool prefix ## _store__(const void * void_arg , buffer_type * buffer , int report_step , bool internal_state) {  \
   const prefix ## _type * arg = prefix ## _safe_cast_const( void_arg ); \
   return prefix ## _store(arg , buffer , report_step , internal_state);        \
}


#define VOID_LOAD(prefix)                                                          \
void prefix ## _load__(void * void_arg , buffer_type * buffer , int report_step) { \
   prefix ## _type * arg = prefix ## _safe_cast( void_arg );                       \
   prefix ## _load(arg , buffer , report_step);                                     \
}

#define VOID_STORE_HEADER(prefix) bool prefix ## _store__(const void * , buffer_type * , int , bool);
#define VOID_LOAD_HEADER(prefix) void prefix ## _load__(void * , buffer_type * , int);

#define VOID_FLOAD(prefix)                                                         \
void prefix ## _fload__(void * void_arg , const char * filename) {                 \
   prefix ## _type * arg = prefix ## _safe_cast( void_arg );                       \
   prefix ## _fload(arg , filename);                                               \
}
#define VOID_FLOAD_HEADER(prefix) void prefix ## _fload__(void * , const char * );


/*****************************************************************/

#define VOID_ECL_WRITE(prefix) \
void prefix ## _ecl_write__(const void * void_arg , const char * path , const char * file , fortio_type * restart_fortio) { \
   const prefix ## _type * arg = prefix ## _safe_cast_const( void_arg );       \
   prefix ## _ecl_write(arg , path , file , restart_fortio);                    \
}

#define VOID_ECL_WRITE_HEADER(prefix) void prefix ## _ecl_write__(const void * , const char * , const char * , fortio_type *);

/*****************************************************************/

#define VOID_ECL_LOAD(prefix) \
bool prefix ## _ecl_load__(void * void_arg , const char * ecl_file  , const ecl_sum_type * ecl_sum, const ecl_file_type * restart_file, int report_step) { \
   prefix ## _type * arg = prefix ## _safe_cast( void_arg );                         \
   return prefix ## _ecl_load(arg , ecl_file , ecl_sum , restart_file , report_step);      \
}

#define VOID_ECL_LOAD_HEADER(prefix) bool prefix ## _ecl_load__(void * , const char * , const ecl_sum_type *, const ecl_file_type * , int);


/*****************************************************************/

#define VOID_FREE(prefix)                        \
void prefix ## _free__(void * void_arg) {         \
   prefix ## _type * arg = prefix ## _safe_cast( void_arg ); \
   prefix ## _free( arg ); \
}

#define VOID_FREE_HEADER(prefix) void prefix ## _free__(void * );


/*****************************************************************/

#define VOID_USER_GET(prefix)                                                     \
double prefix ## _user_get__(void * void_arg , const char * key , bool * valid) { \
   prefix ## _type * arg = prefix ## _safe_cast( void_arg );                      \
   return prefix ## _user_get(arg , key , valid);                                 \
}

#define VOID_USER_GET_HEADER(prefix) double prefix ## _user_get__(void * , const char * , bool *);



/*****************************************************************/

#define VOID_USER_GET_OBS(prefix)                                                     \
void prefix ## _user_get__(void * void_arg , const char * key , double * value, double * std, bool * valid) { \
   prefix ## _user_get((prefix ## _type *) void_arg , key , value , std , valid);               \
}

#define VOID_USER_GET_OBS_HEADER(prefix) void prefix ## _user_get__(void * , const char * , double * , double * , bool *);


/*****************************************************************/

#define VOID_FREE_DATA(prefix)                               \
void prefix ## _free_data__(void * void_arg) {               \
   prefix ## _type * arg = prefix ## _safe_cast( void_arg ); \
   prefix ## _free_data( arg );                              \
}

#define VOID_FREE_DATA_HEADER(prefix) void prefix ## _free_data__(void * );

/*****************************************************************/

/*
#define VOID_REALLOC_DATA(prefix)                            \
void prefix ## _realloc_data__(void * void_arg) {            \
   prefix ## _type * arg = prefix ## _safe_cast( void_arg ); \
   prefix ## _realloc_data( arg );                           \
}

#define VOID_REALLOC_DATA_HEADER(prefix) void prefix ## _realloc_data__(void * );
*/

/*****************************************************************/

#define VOID_COPY(prefix)                                             \
void prefix ## _copy__(const void * void_src, void * void_target) {   \
   const prefix ## _type * src = prefix ## _safe_cast_const( void_src );    \
   prefix ## _type * target = prefix ## _safe_cast( void_target );    \
   prefix ## _copy( src , target );                                   \
}
#define VOID_COPY_HEADER(prefix) void prefix ## _copy__(const void * , void * );

/*****************************************************************/


#define CONFIG_GET_ECL_KW_NAME(prefix)        const char * prefix ## _config_get_ecl_kw_name(const prefix ## _config_type * config) { return config->ecl_kw_name; }
#define CONFIG_GET_ECL_KW_NAME_HEADER(prefix) const char * prefix ## _config_get_ecl_kw_name(const prefix ## _config_type * )


/*****************************************************************/

#define VOID_SERIALIZE(prefix)     \
void prefix ## _serialize__(const void *void_arg, const active_list_type * active_list , matrix_type * A , int row_offset , int column) {\
   const prefix ## _type  *arg = prefix ## _safe_cast_const( void_arg );                                                              \
   prefix ## _serialize (arg , active_list , A , row_offset , column);                                                \
}
#define VOID_SERIALIZE_HEADER(prefix) void prefix ## _serialize__(const void * , const active_list_type * , matrix_type *  , int , int);


#define VOID_DESERIALIZE(prefix)     \
void prefix ## _deserialize__(void *void_arg, const active_list_type * active_list , const matrix_type * A , int row_offset , int column) {\
   prefix ## _type  *arg = prefix ## _safe_cast( void_arg );                                                          \
   prefix ## _deserialize (arg , active_list , A , row_offset , column);                                                      \
}
#define VOID_DESERIALIZE_HEADER(prefix) void prefix ## _deserialize__(void * , const active_list_type * , const matrix_type *  , int , int);




/*****************************************************************/

#define VOID_INITIALIZE(prefix)     \
  bool prefix ## _initialize__(void *void_arg, int iens , mzran_type * rng) {     \
   prefix ## _type  *arg = prefix ## _safe_cast(void_arg);       \
   return prefix ## _initialize (arg , iens , rng);                  \
}
#define VOID_INITIALIZE_HEADER(prefix) bool prefix ## _initialize__(void *, int , mzran_type * );

/*****************************************************************/

#define VOID_SET_INFLATION(prefix) \
void prefix ## _set_inflation__( void * void_inflation , const void * void_std , const void * void_min_std) {                                               \
   prefix ## _set_inflation( prefix ## _safe_cast( void_inflation ) , prefix ## _safe_cast_const( void_std ) , prefix ## _safe_cast_const( void_min_std )); \
}
#define VOID_SET_INFLATION_HEADER(prefix) void prefix ## _set_inflation__( void * void_inflation , const void * void_std , const void * void_min_std );


/*****************************************************************/

#define VOID_GET_OBS(prefix)   \
void prefix ## _get_observations__(const void * void_arg , obs_data_type * obs_data, int report_step , const active_list_type * __active_list) { \
  prefix ## _get_observations((prefix ## _type *) void_arg , obs_data , report_step , __active_list); \
}

#define VOID_GET_OBS_HEADER(prefix) void prefix ## _get_observations__(const void * , obs_data_type * , int , const active_list_type * )

/*****************************************************************/

#define VOID_MEASURE(obs_prefix, state_prefix) \
void obs_prefix ## _measure__(const void * void_obs ,  const void * void_state , int report_step , int iens , meas_data_type * meas_data , const active_list_type * __active_list) { \
   const obs_prefix ## _type   * obs   = obs_prefix ## _safe_cast_const( void_obs );     \
   const state_prefix ## _type * state = state_prefix ## _safe_cast_const( void_state );       \
   obs_prefix ## _measure(obs , state , report_step , iens , meas_data , __active_list);                    \
}

#define VOID_MEASURE_HEADER(obs_prefix) void obs_prefix ## _measure__(const void * ,  const void * , int , int , meas_data_type * , const active_list_type *)


/*****************************************************************/

#define VOID_CHI2(obs_prefix, state_prefix) \
double obs_prefix ## _chi2__(const void * void_obs ,  const void * void_state) {         \
   const obs_prefix ## _type   * obs   = obs_prefix ## _safe_cast_const( void_obs );     \
   const state_prefix ## _type * state = state_prefix ## _safe_cast_const( void_state ); \
   return obs_prefix ## _chi2(obs , state);                                        \
}

#define VOID_CHI2_HEADER(obs_prefix) double obs_prefix ## _chi2__(const void * ,  const void *);


/*****************************************************************/

#define VOID_TRUNCATE(prefix)         void prefix ## _truncate__(void * void_arg) { prefix ## _truncate( (prefix ## _type *) void_arg); }
#define VOID_TRUNCATE_HEADER(prefix)  void prefix ## _truncate__(void * )

/*****************************************************************/
#define VOID_SCALE(prefix)        void prefix ## _scale__(void * void_arg , double scale_factor) { prefix ## _scale( prefix ## _safe_cast( void_arg ) , scale_factor ); }
#define VOID_SCALE_HEADER(prefix) void prefix ## _scale__(void *  , double );

/*****************************************************************/

#define VOID_CLEAR(prefix)         void prefix ## _clear__(void * void_arg) { prefix ## _clear( prefix ## _safe_cast( void_arg )); }
#define VOID_CLEAR_HEADER(prefix)  void prefix ## _clear__(void * )

/*****************************************************************/


#define VOID_ISQRT(prefix)         void prefix ## _isqrt__(void * void_arg) { prefix ## _isqrt( prefix ## _safe_cast( void_arg )); }
#define VOID_ISQRT_HEADER(prefix)  void prefix ## _isqrt__(void * )

/*****************************************************************/

#define VOID_IADD(prefix)   void prefix ## _iadd__( void * void_arg , const void * void_delta ) { \
   prefix ## _iadd( prefix ## _safe_cast( void_arg ) , prefix ## _safe_cast_const( void_delta ) ); \
} 

#define VOID_IADD_HEADER(prefix)   void prefix ## _iadd__( void * void_arg , const void * void_delta );

/*****************************************************************/

#define VOID_IMUL(prefix)   void prefix ## _imul__( void * void_arg , const void * void_delta ) { \
   prefix ## _imul( prefix ## _safe_cast( void_arg ) , prefix ## _safe_cast_const( void_delta ) ); \
} 

#define VOID_IMUL_HEADER(prefix)   void prefix ## _imul__( void * void_arg , const void * void_delta );

/*****************************************************************/

#define VOID_IADDSQR(prefix)   void prefix ## _iaddsqr__( void * void_arg , const void * void_delta ) { \
   prefix ## _iaddsqr( prefix ## _safe_cast( void_arg ) , prefix ## _safe_cast_const( void_delta ) ); \
} 

#define VOID_IADDSQR_HEADER(prefix)   void prefix ## _iaddsqr__( void * void_arg , const void * void_delta );

/*****************************************************************/

#define CONFIG_GET_ENSFILE(prefix)                   const char * prefix ## _config_get_ensfile_ref(const prefix ## _config_type * config) { return config->ensfile; }
#define CONFIG_GET_ECLFILE(prefix)                   const char * prefix ## _config_get_eclfile_ref(const prefix ## _config_type * config) { return config->eclfile; }
#define CONFIG_GET_ENSFILE_HEADER(prefix)       const char * prefix ## _config_get_ensfile_ref(const prefix ## _config_type * )
#define CONFIG_GET_ECLFILE_HEADER(prefix)       const char * prefix ## _config_get_eclfile_ref(const prefix ## _config_type * )

/*****************************************************************/

#define VOID_IGET(prefix)        double prefix ## _iget__(const void * void_arg, int index) { return prefix ## _iget((const prefix ## _type *) void_arg , index); }
#define VOID_IGET_HEADER(prefix) double prefix ## _iget__(const void * , int ) 

#ifdef __cplusplus
}
#endif
#endif
