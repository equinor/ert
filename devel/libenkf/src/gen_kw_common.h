#ifndef __GEN_KW_COMMON_H__
#define __GEN_KW_COMMON_H__

/*
  Contains some headers which both gen_kw.c and gen_kw_config.c need -
  split like this to avoid circular dependencies.
*/


typedef struct gen_kw_config_struct gen_kw_config_type;
typedef struct gen_kw_struct        gen_kw_type;

gen_kw_type * gen_kw_alloc(const gen_kw_config_type * );
void          gen_kw_fload(gen_kw_type * , const char *);



#endif
