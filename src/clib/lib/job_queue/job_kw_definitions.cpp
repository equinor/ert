

#include "ert/config/config_schema_item.hpp"
#include <ert/config/config_parser.hpp>
#include <ert/python.hpp>

#include <ert/job_queue/job_kw_definitions.hpp>

config_item_types job_kw_get_type(const char *arg_type) {

    config_item_types type = CONFIG_INVALID;

    if (strcmp(arg_type, JOB_STRING_TYPE) == 0)
        type = CONFIG_STRING;
    else if (strcmp(arg_type, JOB_INT_TYPE) == 0)
        type = CONFIG_INT;
    else if (strcmp(arg_type, JOB_FLOAT_TYPE) == 0)
        type = CONFIG_FLOAT;
    else if (strcmp(arg_type, JOB_BOOL_TYPE) == 0)
        type = CONFIG_BOOL;
    else if (strcmp(arg_type, JOB_RUNTIME_FILE_TYPE) == 0)
        type = CONFIG_RUNTIME_FILE;
    else if (strcmp(arg_type, JOB_RUNTIME_INT_TYPE) == 0)
        type = CONFIG_RUNTIME_INT;

    return type;
}

const char *kw_from_type(config_item_types typ) {
    switch (typ) {
    case CONFIG_STRING:
    case CONFIG_PATH:
    case CONFIG_EXISTING_PATH:
    case CONFIG_EXECUTABLE:
        return JOB_STRING_TYPE;
    case CONFIG_INT:
        return JOB_INT_TYPE;
    case CONFIG_FLOAT:
        return JOB_FLOAT_TYPE;
    case CONFIG_BOOL:
        return JOB_BOOL_TYPE;
    case CONFIG_RUNTIME_FILE:
        return JOB_RUNTIME_FILE_TYPE;
    case CONFIG_RUNTIME_INT:
        return JOB_RUNTIME_INT_TYPE;
    default:
        return JOB_STRING_TYPE;
    }
}

ERT_CLIB_SUBMODULE("job_kw", m) {
    using namespace py::literals;
    m.def("type_from_kw",
          [](const char *arg_type) { return int(job_kw_get_type(arg_type)); });
    m.def("kw_from_type", [](int typ) {
        return kw_from_type(static_cast<config_item_types>(typ));
    });
}
