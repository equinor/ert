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

void relperm_initialize(relperm_type * relperm, int iens) {
  DEBUG_ASSERT(relperm)
  scalar_sample(relperm->scalar);

}

void relperm_get_data(const relperm_type * relperm, double * data) {
  scalar_get_data(relperm->scalar,data);
}

void relperm_set_data(const relperm_type * relperm, const double * data){
  scalar_set_data(relperm->scalar,data);
}

void relperm_ecl_write(const relperm_type * relperm, const double * data, const char * path){
  DEBUG_ASSERT(relperm)

    {
      /*
      FILE * stream = enkf_util_fopen_w(eclfile,__func__);
      */
      relperm_set_data(relperm,data);
      relperm_output_transform(relperm);
      relperm_config_ecl_write(relperm->config, relperm_get_output_ref(relperm),path); 
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


void relperm_make_tab(relperm_type * relperm){
  int i;
  int size;
  double * output_data; 

  const relperm_config_type * relperm_config = relperm->config;
  size = relperm_config_get_data_size(relperm_config);  
  output_data = malloc(size * sizeof *output_data);
  
  relperm_get_output_data(relperm, output_data);
  /*
  for(i = 0; i <= relperm_config->nsw-1; i++) {
        relperm->swof[i] =  output_data[0] + ((1-output_data[0])/(relperm_config->nsw -1))*(i);

  }
  */
  free(output_data);
}

/* Construct the total water saturation table*/
void relperm_tab_tot_water_sat(relperm_type * relperm) {
 
  /*  printf("Size of relperm->swof %d",size(relperm->swof),__func__); */
  /*  for (i =0 ; i < 5 ; i++){
      } 
  
  relperm->swof[0] = 2.1;
  */

}
