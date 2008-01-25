#ifndef __ENKF_MACROS_H__
#define __ENKF_MACROS_H__

#include <stdio.h>
#include <stdlib.h>
#include <enkf_types.h>

#define CONFIG_STD_FIELDS \
int data_size;            \
char * ecl_kw_name;       \
enkf_var_type var_type;



#define VOID_CONFIG_FREE(prefix)            void prefix ## _config_free__(void *void_arg) { prefix ## _config_free((prefix ## _config_type *) void_arg); }
#define VOID_CONFIG_FREE_HEADER(prefix)     void prefix ## _config_free__(void *)

/*****************************************************************/
/*
#define GET_SERIAL_SIZE(prefix)                                              \
int prefix ## _config_get_serial_size (const prefix ## _config_type *arg) {  \
   return arg->serial_size;                                                  \
}
#define GET_SERIAL_SIZE_HEADER(prefix)      int prefix ## _config_get_serial_size (const prefix ## _config_type *)
#define VOID_GET_SERIAL_SIZE(prefix)        int prefix ## _config_get_serial_size__ (const void *void_arg ) { return prefix ## _config_get_serial_size((const prefix ## _config_type *) void_arg); }
#define VOID_GET_SERIAL_SIZE_HEADER(prefix) int prefix ## _config_get_serial_size__ (const void *)
*/

/*****************************************************************/

#define GET_DATA_SIZE(prefix)               int prefix ## _config_get_data_size (const prefix ## _config_type *arg) { return arg->data_size; }
#define GET_DATA_SIZE_HEADER(prefix)        int prefix ## _config_get_data_size (const prefix ## _config_type *)
/*
  #define VOID_GET_DATA_SIZE(prefix)          int prefix ## _config_get_data_size__ (const void *void_arg) { return prefix ## _config_get_data_size((const prefix ## _config_type *) void_arg); }
  #define VOID_GET_DATA_SIZE_HEADER(prefix)   int prefix ## _config_get_data_size__ (const void *)
*/


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
void prefix ## _fwrite__(const void * void_arg , FILE * stream) { \
   prefix ## _fwrite((const prefix ## _type *) void_arg , stream);      \
}

#define VOID_FREAD(prefix) \
void prefix ## _fread__(void * void_arg , FILE * stream) { \
   prefix ## _fread((prefix ## _type *) void_arg , stream);      \
}

#define VOID_FWRITE_HEADER(prefix) void prefix ## _fwrite__(const void * , FILE *);
#define VOID_FREAD_HEADER(prefix) void prefix ## _fread__(void * , FILE *);


/*****************************************************************/

#define VOID_ECL_WRITE(prefix) \
void prefix ## _ecl_write__(const void * void_arg , const char * path) { \
   prefix ## _ecl_write((const prefix ## _type *) void_arg , path);      \
}

#define VOID_ECL_READ(prefix) \
void prefix ## _ecl_read__(void * void_arg , const char * path) { \
   prefix ## _ecl_read((prefix ## _type *) void_arg , path);      \
}

#define VOID_ECL_WRITE_HEADER(prefix) void prefix ## _ecl_write__(const void * , const char * );
#define VOID_ECL_READ_HEADER(prefix) void prefix ## _ecl_read__(void * , const char * );


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

#define VOID_SWAPIN(prefix) \
void prefix ## _swapin__(void *void_arg , FILE * stream) { \
   prefix ## _swapin((prefix ## _type *) void_arg , stream);     \
}                                                            

#define VOID_SWAPOUT(prefix) \
void prefix ## _swapout__(void *void_arg , FILE * stream) { \
   prefix ## _swapout((prefix ## _type *) void_arg , stream);     \
}                                                            


#define VOID_SWAPIN_HEADER(prefix)  void prefix ## _swapin__(void *  , FILE * );
#define VOID_SWAPOUT_HEADER(prefix) void prefix ## _swapout__(void * , FILE * );

/*****************************************************************/

#define CONFIG_GET_ECL_KW_NAME(prefix)        const char * prefix ## _config_get_ecl_kw_name(const prefix ## _config_type * config) { return config->ecl_kw_name; }
#define CONFIG_GET_ECL_KW_NAME_HEADER(prefix) const char * prefix ## _config_get_ecl_kw_name(const prefix ## _config_type * )


/*****************************************************************/
#define VOID_SERIALIZE(prefix)     \
int prefix ## _serialize__(const void *void_arg, int internal_offset , size_t serial_data_size , double *serial_data , size_t stride , size_t offset , bool *complete) { \
   const prefix ## _type  *arg = (const prefix ## _type *) void_arg;       \
   return prefix ## _serialize (arg , internal_offset , serial_data_size , serial_data , stride , offset , complete);       \
}
#define VOID_SERIALIZE_HEADER(prefix) int prefix ## _serialize__(const void *, int , size_t , double *, size_t , size_t , bool *);


#define VOID_DESERIALIZE(prefix)     \
int prefix ## _deserialize__(void *void_arg, int internal_offset , size_t serial_size , const double *serial_data , size_t stride , size_t offset) { \
   prefix ## _type  *arg = (prefix ## _type *) void_arg;       \
   return prefix ## _deserialize (arg , internal_offset , serial_size , serial_data , stride , offset);       \
}
#define VOID_DESERIALIZE_HEADER(prefix) int prefix ## _deserialize__(void *, int , size_t , const double *, size_t , size_t);


/*****************************************************************/

#define VOID_GET_OBS(prefix)   \
void prefix ## _get_observations__(const void * void_arg , int report_step, obs_data_type * obs_data) {   \
   prefix ## _get_observations((prefix ## _type *) void_arg , report_step , obs_data); \
}

#define VOID_GET_OBS_HEADER(prefix) void prefix ## _get_observations__(const void * , int , obs_data_type *)

/*****************************************************************/

#define VOID_MEASURE(prefix)   \
void prefix ## _obs_measure__(const void * void_arg ,  const void * domain_object , meas_vector_type * meas_vector) {         \
   prefix ## _obs_measure((const prefix ## _obs_type *) void_arg , (const prefix ## _type  * ) domain_object , meas_vector); \
}

#define VOID_MEASURE_HEADER(prefix) void prefix ## _obs_measure__(const void * ,  const void * , meas_vector_type *)


/*****************************************************************/

#define VOID_TRUNCATE(prefix)         void prefix ## _truncate__(void * void_arg) { prefix ## _truncate( (prefix ## _type *) void_arg); }
#define VOID_TRUNCATE_HEADER(prefix)  void prefix ## _truncate__(void * )

/*****************************************************************/

#define VOID_FREE_CONFIG(prefix)        void prefix ## _config_free__(void * void_arg) { prefix ## _config_free( (prefix ## _config_type *) void_arg); }
#define VOID_FREE_CONFIG_HEADER(prefix) void prefix ## _config_free__(void * )

/*****************************************************************/

#define CONFIG_GET_ENSFILE(prefix)       	     const char * prefix ## _config_get_ensfile_ref(const prefix ## _config_type * config) { return config->ensfile; }
#define CONFIG_GET_ECLFILE(prefix)       	     const char * prefix ## _config_get_eclfile_ref(const prefix ## _config_type * config) { return config->eclfile; }
#define CONFIG_GET_ENSFILE_HEADER(prefix)       const char * prefix ## _config_get_ensfile_ref(const prefix ## _config_type * )
#define CONFIG_GET_ECLFILE_HEADER(prefix)       const char * prefix ## _config_get_eclfile_ref(const prefix ## _config_type * )



#endif
