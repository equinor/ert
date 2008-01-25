#ifndef __GEN_KW_CONFIG_H__
#define __GEN_KW_CONFIG_H__

#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <logmode.h>
#include <scalar_config.h>
#include <scalar.h>

typedef struct gen_kw_config_struct gen_kw_config_type;

struct gen_kw_config_struct {
  char               ** kw_list;
  enkf_var_type         var_type;  
  scalar_config_type  * scalar_config;
  char                * template_file;
};


const char          * gen_kw_config_get_template_ref(const gen_kw_config_type * );
gen_kw_config_type  * gen_kw_config_fscanf_alloc(const char * , const char *);
void                  gen_kw_config_free(gen_kw_config_type *);
void                  gen_kw_config_transform(const gen_kw_config_type * , const double * , double *);
void                  gen_kw_config_truncate(const gen_kw_config_type * , scalar_type * );
int                   gen_kw_config_get_data_size(const gen_kw_config_type * );
const char          * gen_kw_config_get_name(const gen_kw_config_type * , int );

VOID_FUNC_HEADER(gen_kw_config_free);
#endif
