#ifndef __GEN_DATA_COMMON_H__
#define __GEN_DATA_COMMON_H__

/*
  Contains some headers which both gen_data.c and gen_data_config.c need -
  split like this to avoid circular dependencies.
*/


typedef struct gen_data_config_struct gen_data_config_type;
typedef struct gen_data_struct        gen_data_type;

gen_data_type * gen_data_alloc(const gen_data_config_type * );
void            gen_data_fload(gen_data_type * , const char * );



#endif
