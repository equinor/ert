#ifndef __ENKF_MACROS_H__
#define __ENKF_MACROS_H__

#include <stdio.h>
#include <stdlib.h>
#include <enkf_types.h>
#include <ecl_block.h>
#include <ecl_sum.h>
#include <enkf_serialize.h>

#define CONFIG_STD_FIELDS \
int data_size;            \
char * ecl_kw_name;       \
enkf_var_type var_type;



#define VOID_CONFIG_FREE(prefix)            void prefix ## _config_free__(void *void_arg) { prefix ## _config_free((prefix ## _config_type *) void_arg); }
#define VOID_CONFIG_FREE_HEADER(prefix)     void prefix ## _config_free__(void *)

/*****************************************************************/

#define GET_DATA_SIZE(prefix)               int prefix ## _config_get_data_size (const prefix ## _config_type *arg) { return arg->data_size; }
#define GET_DATA_SIZE_HEADER(prefix)        int prefix ## _config_get_data_size (const prefix ## _config_type *)

/*****************************************************************/


#define CONFIG_SET_ECLFILE(prefix)                                                            \
void prefix ## _config_set_eclfile(prefix ## _config_type *config , const char * file) {      \
  if (file != NULL) {                                                                         \
     config->eclfile = realloc(config->eclfile , strlen(file) + 1);                           \
     strcpy(config->eclfile , file);                                                          \
  } else config->eclfile = NULL; 							      \
}


#define CONFIG_SET_ENSFILE(prefix)                                                           \
void prefix ## _config_set_ensfile(prefix ## _config_type *config , const char * file) {     \
  if (file != NULL) {                                                                        \
     config->ensfile = realloc(config->ensfile , strlen(file) + 1);                          \
     strcpy(config->ensfile , file);                                                         \
  } else config->ensfile = NULL;                                                             \
}


#define CONFIG_SET_ENSFILE_VOID(prefix)                                            \
void prefix ## _config_set_ensfile__(void *void_config , const char * file) {       \
   prefix ## _config_type * config = (prefix ## _config_type *) void_config;        \
   prefix ## _config_set_ensfile(config , file);                                    \
}

#define CONFIG_SET_ECLFILE_VOID(prefix)                                            \
void prefix ## _config_set_eclfile__(void *void_config , const char * file) {       \
   prefix ## _config_type * config = (prefix ## _config_type *) void_config;        \
   prefix ## _config_set_eclfile(config , file);                                    \
}

/*****************************************************************/

#define CONFIG_SET_ECLFILE_HEADER(prefix) 	void prefix ## _config_set_eclfile  (prefix ## _config_type *, const char * );
#define CONFIG_SET_ENSFILE_HEADER(prefix) 	void prefix ## _config_set_ensfile  (prefix ## _config_type *, const char * );
#define CONFIG_SET_ECLFILE_HEADER_VOID(prefix) void prefix ## _config_set_eclfile__(void *, const char * );
#define CONFIG_SET_ENSFILE_HEADER_VOID(prefix) void prefix ## _config_set_ensfile__(void *, const char * );


/*****************************************************************/

#define VOID_ALLOC(prefix) \
void * prefix ## _alloc__(const void *void_config) {                      \
  return prefix ## _alloc((const prefix ## _config_type *) void_config);  \
}

#define VOID_ALLOC_HEADER(prefix) void * prefix ## _alloc__(const void *)


/*****************************************************************/

#define VOID_FWRITE(prefix) \
bool prefix ## _fwrite__(const void * void_arg , FILE * stream) { \
   return prefix ## _fwrite((const prefix ## _type *) void_arg , stream);      \
}

#define VOID_FREAD(prefix) \
void prefix ## _fread__(void * void_arg , FILE * stream) { \
   prefix ## _fread((prefix ## _type *) void_arg , stream);      \
}

#define VOID_FWRITE_HEADER(prefix) bool prefix ## _fwrite__(const void * , FILE *);
#define VOID_FREAD_HEADER(prefix) void prefix ## _fread__(void * , FILE *);


/*****************************************************************/

#define VOID_ECL_WRITE(prefix) \
void prefix ## _ecl_write__(const void * void_arg , const char * path) { \
   prefix ## _ecl_write((const prefix ## _type *) void_arg , path);      \
}

#define VOID_ECL_WRITE_HEADER(prefix) void prefix ## _ecl_write__(const void * , const char * );

/*****************************************************************/

#define VOID_ECL_LOAD(prefix) \
void prefix ## _ecl_load__(void * void_arg , const char * ecl_file  , const ecl_sum_type * ecl_sum, const ecl_block_type * restart_block, int report_step) { \
   prefix ## _ecl_load((prefix ## _type *) void_arg , ecl_file , ecl_sum , restart_block , report_step);      \
}

#define VOID_ECL_LOAD_HEADER(prefix) void prefix ## _ecl_load__(void * , const char * , const ecl_sum_type *, const ecl_block_type * , int);


/*****************************************************************/

#define VOID_FREE(prefix)                        \
void prefix ## _free__(void * void_arg) {         \
   prefix ## _free((prefix ## _type *) void_arg); \
}

#define VOID_FREE_HEADER(prefix) void prefix ## _free__(void * );


/*****************************************************************/

#define VOID_FREE_DATA(prefix)                        \
void prefix ## _free_data__(void * void_arg) {         \
   prefix ## _free_data((prefix ## _type *) void_arg); \
}

#define VOID_FREE_DATA_HEADER(prefix) void prefix ## _free_data__(void * );

/*****************************************************************/

#define VOID_REALLOC_DATA(prefix)                        \
void prefix ## _realloc_data__(void * void_arg) {         \
   prefix ## _realloc_data((prefix ## _type *) void_arg); \
}

#define VOID_REALLOC_DATA_HEADER(prefix) void prefix ## _realloc_data__(void * );

/*****************************************************************/

#define VOID_COPYC(prefix)                                      \
void * prefix ## _copyc__(const void * void_arg) {    \
   return prefix ## _copyc((const prefix ## _type *) void_arg); \
}

#define VOID_COPYC_HEADER(prefix) void * prefix ## _copyc__(const void * )

/*****************************************************************/


#define CONFIG_GET_ECL_KW_NAME(prefix)        const char * prefix ## _config_get_ecl_kw_name(const prefix ## _config_type * config) { return config->ecl_kw_name; }
#define CONFIG_GET_ECL_KW_NAME_HEADER(prefix) const char * prefix ## _config_get_ecl_kw_name(const prefix ## _config_type * )


/*****************************************************************/
#define VOID_SERIALIZE(prefix)     \
int prefix ## _serialize__(const void *void_arg, serial_state_type * serial_state , size_t offset , serial_vector_type * serial_vector) { \
   const prefix ## _type  *arg = (const prefix ## _type *) void_arg;       \
   return prefix ## _serialize (arg , serial_state , offset , serial_vector);       \
}
#define VOID_SERIALIZE_HEADER(prefix) int prefix ## _serialize__(const void *, serial_state_type * , size_t , serial_vector_type *);


#define VOID_DESERIALIZE(prefix)     \
void prefix ## _deserialize__(void *void_arg, serial_state_type * serial_state, const serial_vector_type * serial_vector) { \
   prefix ## _type  *arg = (prefix ## _type *) void_arg;       \
   prefix ## _deserialize (arg , serial_state , serial_vector); \
}
#define VOID_DESERIALIZE_HEADER(prefix) void prefix ## _deserialize__(void *, serial_state_type * , const serial_vector_type *);

/*****************************************************************/
#define VOID_INITIALIZE(prefix)     \
void prefix ## _initialize__(void *void_arg, int iens) {              \
   prefix ## _type  *arg = (prefix ## _type *) void_arg;       \
   prefix ## _initialize (arg , iens);                              \
}
#define VOID_INITIALIZE_HEADER(prefix) void prefix ## _initialize__(void *, int);




/*****************************************************************/

#define VOID_GET_OBS(prefix)   \
void prefix ## _get_observations__(const void * void_arg , int report_step, obs_data_type * obs_data) {   \
   prefix ## _get_observations((prefix ## _type *) void_arg , report_step , obs_data); \
}

#define VOID_GET_OBS_HEADER(prefix) void prefix ## _get_observations__(const void * , int , obs_data_type *)

/*****************************************************************/

#define VOID_MEASURE(obs_prefix, state_prefix) \
void obs_prefix ## _measure__(const void * void_arg ,  const void * state_object , meas_vector_type * meas_vector) {         \
   obs_prefix ## _measure((const obs_prefix ## _type *) void_arg , (const state_prefix ## _type  * ) state_object , meas_vector); \
}

#define VOID_MEASURE_HEADER(obs_prefix) void obs_prefix ## _measure__(const void * ,  const void * , meas_vector_type *)


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

#define ASSERT_TYPE(prefix) \
void prefix ## _assert_type(const prefix ## _type * object) { \
  if (object->__impl_type != TARGET_TYPE)  \
    util_abort("%s: assert_type failed\n",__func__); \
}
#define ASSERT_TYPE_HEADER(prefix) void prefix ## _assert_type(const prefix ## _type * );


/*****************************************************************/

#define VOID_IGET(prefix)        double prefix ## _iget__(const void * void_arg, int index) { return prefix ## _iget((const prefix ## _type *) void_arg , index); }
#define VOID_IGET_HEADER(prefix) double prefix ## _iget__(const void * , int ) 

#endif
