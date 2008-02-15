#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <util.h>
#include <enkf_types.h>
#include <relperm_config.h>
#include <relperm.h>
#include <enkf_util.h>
#include <scalar.h>


#define  TARGET_TYPE RELPERM
#define  DEBUG
#include "enkf_debug.h"

/*****************************************************************/
struct relperm_struct{
  DEBUG_DECLARE
  const relperm_config_type  * config;
  scalar_type                * scalar;   
};

/*****************************************************************/


relperm_type * relperm_alloc(const relperm_config_type * relperm_config){
  relperm_type * relperm = malloc(sizeof *relperm);
  relperm->config = relperm_config;
  relperm->scalar = scalar_alloc(relperm_config->scalar_config);
  DEBUG_ASSIGN(relperm);
  return relperm;
}

void relperm_initialize(relperm_type * relperm) {
  scalar_sample(relperm->scalar);
  DEBUG_ASSERT(relperm)
}

void relperm_get_data(const relperm_type * relperm, double * data) {
  scalar_get_data(relperm->scalar,data);
}

void relperm_set_data(relperm_type * relperm, const double * data){
  scalar_set_data(relperm->scalar,data);
}

void relperm_ecl_write(const relperm_type * relperm, const char * eclfile){
  DEBUG_ASSERT(relperm)

    {
      FILE * stream = enkf_util_fopen_w(eclfile,__func__);
      relperm_output_transform(relperm);
      relperm_config_ecl_write(relperm->config, relperm_get_output_ref(relperm),stream); 
      fclose(stream);
    }
}

void relperm_output_transform(const relperm_type * relperm){
  scalar_transform(relperm->scalar);
}

const double * relperm_get_output_ref(const relperm_type * relperm){
  return scalar_get_output_ref(relperm->scalar) ;
}

void relperm_get_output_data(const relperm_type * relperm, double * output_data){
  scalar_get_output_data(relperm->scalar,output_data);
}
