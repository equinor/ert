#ifndef __GEN_DATA_ACTIVE_H__
#define __GEN_DATA_ACTIVE_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <enkf_macros.h>

typedef struct gen_data_active_struct gen_data_active_type;

void gen_data_active_update_active_list(const gen_data_active_type * gen_ , active_list_type * );

SAFE_CAST_HEADER(gen_data_active);

#ifdef __cplusplus
}
#endif
#endif
