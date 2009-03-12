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

#define CONFIG_STD_FIELDS \
int __type_id;            \
int data_size;            \
char * ecl_kw_name;       


/*****************************************************************/
#define GET_ACTIVE_LIST(prefix)        const active_list_type * prefix ## _config_get_active_list(const prefix ## _config_type * config) { return config->active_list; } 
#define GET_ACTIVE_LIST_HEADER(prefix) const active_list_type * prefix ## _config_get_active_list(const prefix ## _config_type * );


/*****************************************************************/
#define IS_INSTANCE(prefix,ID) \
bool prefix ## _is_instance__(const void * __arg) {             	   \
  prefix ## _type * arg = (prefix ## _type *) __arg;         	   \
  if (arg->__type_id != ID)                                        \
     return false;                                                 \
  else                                                             \
     return true;                                                  \
}

#define IS_INSTANCE_HEADER(prefix)  bool prefix ## _is_instance__(const void * );

/******************************************************************/

#define SAFE_CAST(prefix , ID) \
prefix ## _type * prefix ## _safe_cast(const void * __arg) {   \
  prefix ## _type * arg = (prefix ## _type *) __arg;         \
  if (arg->__type_id != ID)                                     \
    util_abort("%s: run_time cast failed: got:%d  expected:%d  - aborting \n",__func__ , arg->__type_id , ID); \
  return arg;                                                   \
}

#define SAFE_CAST_HEADER(prefix) prefix ## _type * prefix ## _safe_cast(const void * );



/*****************************************************************/

#define VOID_CONFIG_ACTIVATE(prefix) \
void prefix ## _config_activate__(void * void_config , active_mode_type active_mode , void * active_info) { \
    prefix ## _config_type * config = prefix ## _config_safe_cast( void_config );                           \
    prefix ## _config_activate( config , active_mode , active_info);                                        \
}

#define VOID_CONFIG_ACTIVATE_HEADER(prefix) void prefix ## _config_activate__(void * , active_mode_type , void *);

/*****************************************************************/

#define VOID_OBS_ACTIVATE(prefix) \
void prefix ## _activate__(void * void_obs , active_mode_type active_mode , void * active_info) { \
    prefix ## _type * obs = prefix ## _safe_cast( void_obs );                           \
    prefix ## _activate( obs , active_mode , active_info);                                        \
}

#define VOID_OBS_ACTIVATE_HEADER(prefix) void prefix ## _activate__(void * , active_mode_type , void *);

/*****************************************************************/

#define VOID_CONFIG_FREE(prefix)            void prefix ## _config_free__(void *void_arg) { prefix ## _config_free((prefix ## _config_type *) void_arg); }
#define VOID_CONFIG_FREE_HEADER(prefix)     void prefix ## _config_free__(void *)

/*****************************************************************/

#define GET_DATA_SIZE(prefix)               int prefix ## _config_get_data_size (const prefix ## _config_type *arg) { return arg->data_size; }
#define GET_DATA_SIZE_HEADER(prefix)        int prefix ## _config_get_data_size (const prefix ## _config_type *)


/*****************************************************************/

#define VOID_ALLOC(prefix)                                                            \
void * prefix ## _alloc__(const void *void_config) {                                  \
  const prefix ## _config_type * config = prefix ## _config_safe_cast( void_config ); \
  return prefix ## _alloc(config);                                                    \
}

#define VOID_ALLOC_HEADER(prefix) void * prefix ## _alloc__(const void *)

/*****************************************************************/

#define VOID_FWRITE(prefix)                                        \
bool prefix ## _fwrite__(const void * void_arg , FILE * stream , bool internal_state) {  \
   const prefix ## _type * arg = prefix ## _safe_cast( void_arg ); \
   return prefix ## _fwrite(arg , stream , internal_state);        \
}


#define VOID_FREAD(prefix)                                  \
void prefix ## _fread__(void * void_arg , FILE * stream) {  \
   prefix ## _type * arg = prefix ## _safe_cast( void_arg );\
   prefix ## _fread(arg , stream);                          \
}

#define VOID_FWRITE_HEADER(prefix) bool prefix ## _fwrite__(const void * , FILE * , bool);
#define VOID_FREAD_HEADER(prefix) void prefix ## _fread__(void * , FILE *);


/*****************************************************************/

#define VOID_ECL_WRITE(prefix) \
void prefix ## _ecl_write__(const void * void_arg , const char * path , const char * file , fortio_type * restart_fortio) { \
   const prefix ## _type * arg = prefix ## _safe_cast( void_arg );       \
   prefix ## _ecl_write(arg , path , file , restart_fortio);                    \
}

#define VOID_ECL_WRITE_HEADER(prefix) void prefix ## _ecl_write__(const void * , const char * , const char * , fortio_type *);

/*****************************************************************/

#define VOID_ECL_LOAD(prefix) \
void prefix ## _ecl_load__(void * void_arg , const char * ecl_file  , const ecl_sum_type * ecl_sum, const ecl_file_type * restart_file, int report_step) { \
   prefix ## _type * arg = prefix ## _safe_cast( void_arg );                         \
   prefix ## _ecl_load(arg , ecl_file , ecl_sum , restart_file , report_step);      \
}

#define VOID_ECL_LOAD_HEADER(prefix) void prefix ## _ecl_load__(void * , const char * , const ecl_sum_type *, const ecl_file_type * , int);


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
   prefix ## _type * arg = prefix ## _safe_cast( void_arg );    		  \
   return prefix ## _user_get(arg , key , valid);               		  \
}

#define VOID_USER_GET_HEADER(prefix) double prefix ## _user_get__(void * , const char * , bool *);



/*****************************************************************/

#define VOID_USER_GET_OBS(prefix)                                                     \
void prefix ## _user_get__(void * void_arg , const char * key , double * value, double * std, bool * valid) { \
   prefix ## _user_get((prefix ## _type *) void_arg , key , value , std , valid);               \
}

#define VOID_USER_GET_OBS_HEADER(prefix) void prefix ## _user_get__(void * , const char * , double * , double * , bool *);


/*****************************************************************/

#define VOID_FREE_DATA(prefix)                         	     \
void prefix ## _free_data__(void * void_arg) {         	     \
   prefix ## _type * arg = prefix ## _safe_cast( void_arg ); \
   prefix ## _free_data( arg );                              \
}

#define VOID_FREE_DATA_HEADER(prefix) void prefix ## _free_data__(void * );

/*****************************************************************/

#define VOID_REALLOC_DATA(prefix)                            \
void prefix ## _realloc_data__(void * void_arg) {            \
   prefix ## _type * arg = prefix ## _safe_cast( void_arg ); \
   prefix ## _realloc_data( arg );                           \
}

#define VOID_REALLOC_DATA_HEADER(prefix) void prefix ## _realloc_data__(void * );

/*****************************************************************/

#define VOID_COPYC(prefix)                                        \
void * prefix ## _copyc__(const void * void_arg) {                \
   const prefix ## _type * arg = prefix ## _safe_cast( void_arg );\
   return prefix ## _copyc( arg );                                \
}

#define VOID_COPYC_HEADER(prefix) void * prefix ## _copyc__(const void * )

/*****************************************************************/


#define CONFIG_GET_ECL_KW_NAME(prefix)        const char * prefix ## _config_get_ecl_kw_name(const prefix ## _config_type * config) { return config->ecl_kw_name; }
#define CONFIG_GET_ECL_KW_NAME_HEADER(prefix) const char * prefix ## _config_get_ecl_kw_name(const prefix ## _config_type * )


/*****************************************************************/
#define VOID_SERIALIZE(prefix)     \
int prefix ## _serialize__(const void *void_arg, serial_state_type * serial_state , size_t offset , serial_vector_type * serial_vector) { \
   const prefix ## _type  *arg = prefix ## _safe_cast( void_arg );       \
   return prefix ## _serialize (arg , serial_state , offset , serial_vector);       \
}
#define VOID_SERIALIZE_HEADER(prefix) int prefix ## _serialize__(const void *, serial_state_type * , size_t , serial_vector_type *);


#define VOID_DESERIALIZE(prefix)     \
void prefix ## _deserialize__(void *void_arg, serial_state_type * serial_state, const serial_vector_type * serial_vector) { \
   prefix ## _type  *arg = prefix ## _safe_cast( void_arg );    \
   prefix ## _deserialize (arg , serial_state , serial_vector); \
}
#define VOID_DESERIALIZE_HEADER(prefix) void prefix ## _deserialize__(void *, serial_state_type * , const serial_vector_type *);


/*****************************************************************/

#define VOID_INITIALIZE(prefix)     \
bool prefix ## _initialize__(void *void_arg, int iens) {         \
   prefix ## _type  *arg = prefix ## _safe_cast(void_arg);       \
   return prefix ## _initialize (arg , iens);                    \
}
#define VOID_INITIALIZE_HEADER(prefix) bool prefix ## _initialize__(void *, int);


/*****************************************************************/

#define VOID_GET_OBS(prefix)   \
void prefix ## _get_observations__(const void * void_arg , int report_step, obs_data_type * obs_data) {   \
   prefix ## _get_observations((prefix ## _type *) void_arg , report_step , obs_data); \
}

#define VOID_GET_OBS_HEADER(prefix) void prefix ## _get_observations__(const void * , int , obs_data_type *)

/*****************************************************************/

#define VOID_MEASURE(obs_prefix, state_prefix) \
void obs_prefix ## _measure__(const void * void_obs ,  const void * void_state , meas_vector_type * meas_vector) {         \
   const obs_prefix ## _type   * obs   = obs_prefix ## _safe_cast( void_obs );     \
   const state_prefix ## _type * state = state_prefix ## _safe_cast( void_state ); \
   obs_prefix ## _measure(obs , state , meas_vector);                              \
}

#define VOID_MEASURE_HEADER(obs_prefix) void obs_prefix ## _measure__(const void * ,  const void * , meas_vector_type *)


/*****************************************************************/

#define VOID_CHI2(obs_prefix, state_prefix) \
double obs_prefix ## _chi2__(const void * void_obs ,  const void * void_state) {   \
   const obs_prefix ## _type   * obs   = obs_prefix ## _safe_cast( void_obs );     \
   const state_prefix ## _type * state = state_prefix ## _safe_cast( void_state ); \
   return obs_prefix ## _chi2(obs , state);                                        \
}

#define VOID_CHI2_HEADER(obs_prefix) double obs_prefix ## _chi2__(const void * ,  const void *);


/*****************************************************************/

#define VOID_TRUNCATE(prefix)         void prefix ## _truncate__(void * void_arg) { prefix ## _truncate( (prefix ## _type *) void_arg); }
#define VOID_TRUNCATE_HEADER(prefix)  void prefix ## _truncate__(void * )

/*****************************************************************/

#define VOID_SCALE(prefix)        void prefix ## _scale(void * void_arg) { prefix ## _scale( (prefix ## _type *) void_arg); }
#define VOID_SCALE_HEADER(prefix) void prefix ## _scale(void * );

/*****************************************************************/

#define VOID_CLEAR(prefix)         void prefix ## _clear__(void * void_arg) { prefix ## _clear( (prefix ## _type *) void_arg); }
#define VOID_CLEAR_HEADER(prefix)  void prefix ## _clear__(void * )

/*****************************************************************/

#define CONFIG_GET_ENSFILE(prefix)       	     const char * prefix ## _config_get_ensfile_ref(const prefix ## _config_type * config) { return config->ensfile; }
#define CONFIG_GET_ECLFILE(prefix)       	     const char * prefix ## _config_get_eclfile_ref(const prefix ## _config_type * config) { return config->eclfile; }
#define CONFIG_GET_ENSFILE_HEADER(prefix)       const char * prefix ## _config_get_ensfile_ref(const prefix ## _config_type * )
#define CONFIG_GET_ECLFILE_HEADER(prefix)       const char * prefix ## _config_get_eclfile_ref(const prefix ## _config_type * )

/*****************************************************************/

#define VOID_FPRINTF_RESULTS(prefix) \
void prefix ## _ensemble_fprintf_results__(const void ** void_ensemble , int ens_size , const char * filename) { \
   const prefix ## _type ** ensemble = (const prefix ## _type **) void_ensemble;                                          \
   prefix ## _ensemble_fprintf_results( ensemble , ens_size , filename) ;                                                  \
}
#define VOID_FPRINTF_RESULTS_HEADER(prefix) void prefix ## _ensemble_fprintf_results__(const void ** , int , const char * );

/*****************************************************************/


#define VOID_IGET(prefix)        double prefix ## _iget__(const void * void_arg, int index) { return prefix ## _iget((const prefix ## _type *) void_arg , index); }
#define VOID_IGET_HEADER(prefix) double prefix ## _iget__(const void * , int ) 

#ifdef __cplusplus
}
#endif
#endif
