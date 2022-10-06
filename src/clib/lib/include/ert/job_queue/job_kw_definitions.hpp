#ifndef ERT_JOB_KW_DEFINITIONS_H
#define ERT_JOB_KW_DEFINITIONS_H

#define MIN_ARG_KEY "MIN_ARG"
#define MAX_ARG_KEY "MAX_ARG"
#define ARG_TYPE_KEY "ARG_TYPE"
#define EXECUTABLE_KEY "EXECUTABLE"

#define JOB_STRING_TYPE "STRING"
#define JOB_INT_TYPE "INT"
#define JOB_FLOAT_TYPE "FLOAT"
#define JOB_BOOL_TYPE "BOOL"
#define JOB_RUNTIME_FILE_TYPE "RUNTIME_FILE"
#define JOB_RUNTIME_INT_TYPE "RUNTIME_INT"

config_item_types job_kw_get_type(const char *arg_type);

#endif
