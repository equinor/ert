#ifndef ERT_GEN_COMMON_H
#define ERT_GEN_COMMON_H

#include <stdio.h>
#include <stdlib.h>

#include <ert/ecl/ecl_type.h>
#include <ert/enkf/gen_data_config.hpp>

void *gen_common_fscanf_alloc(const char *, ecl_data_type, int *);
void *gen_common_fload_alloc(const char *, gen_data_file_format_type,
                             ecl_data_type, ecl_type_enum *, int *);

#endif
