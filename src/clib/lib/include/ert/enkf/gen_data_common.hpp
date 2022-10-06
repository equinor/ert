#ifndef ERT_GEN_DATA_COMMON_H
#define ERT_GEN_DATA_COMMON_H

/*
  Contains some headers which both gen_data.c and gen_data_config.c need -
  split like this to avoid circular dependencies.
*/

typedef struct gen_data_config_struct gen_data_config_type;
typedef struct gen_data_struct gen_data_type;

extern "C" gen_data_type *gen_data_alloc(const gen_data_config_type *);

#endif
