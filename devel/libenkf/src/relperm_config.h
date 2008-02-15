#ifndef __RELPERM_CONFIG_H__
#define __RELPERM_CONFIG_H__


#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <enkf_types.h>
#include <logmode.h>
#include <scalar.h>
#include <scalar_config.h>

typedef struct {
  
  int           * relperm_num1;
  int           * relperm_num2;
  
  enkf_var_type var_type;
  scalar_config_type * scalar_config;
} relperm_config_type;

/*
relperm_config_type * relperm_config_alloc_empty(int);
*/

relperm_config_type * relperm_config_fscanf_alloc(const char *);
void relperm_config_ecl_write(const relperm_config_type *, const double *, FILE *);

/*Generated headers */
GET_DATA_SIZE_HEADER(relperm);
#endif
