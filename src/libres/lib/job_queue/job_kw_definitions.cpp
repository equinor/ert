

#include <ert/config/config_parser.hpp>

#include <ert/job_queue/job_kw_definitions.hpp>


config_item_types job_kw_get_type(const char * arg_type) {

  config_item_types type = CONFIG_INVALID;

  if (strcmp( arg_type , JOB_STRING_TYPE) == 0)
    type = CONFIG_STRING;
  else if (strcmp( arg_type , JOB_INT_TYPE) == 0)
    type = CONFIG_INT;
  else if (strcmp( arg_type , JOB_FLOAT_TYPE) == 0)
    type = CONFIG_FLOAT;
  else if (strcmp( arg_type , JOB_BOOL_TYPE) == 0)
    type = CONFIG_BOOL;
  else if (strcmp( arg_type , JOB_RUNTIME_FILE_TYPE) == 0)
    type = CONFIG_RUNTIME_FILE;
  else if (strcmp( arg_type , JOB_RUNTIME_INT_TYPE) == 0)
    type = CONFIG_RUNTIME_INT;

  return type;
}

