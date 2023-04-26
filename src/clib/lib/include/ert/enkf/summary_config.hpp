#ifndef ERT_SUMMARY_CONFIG_H
#define ERT_SUMMARY_CONFIG_H

#include <stdbool.h>
#include <stdlib.h>

#include <ert/ecl/ecl_smspec.h>
#include <ert/ecl/ecl_sum.h>

#include <ert/enkf/enkf_macros.hpp>

typedef struct summary_config_struct summary_config_type;
typedef struct summary_struct summary_type;

extern "C" summary_config_type *summary_config_alloc(const char *);
extern "C" void summary_config_free(summary_config_type *);

GET_DATA_SIZE_HEADER(summary);
VOID_GET_DATA_SIZE_HEADER(summary);
VOID_CONFIG_FREE_HEADER(summary);

#endif
