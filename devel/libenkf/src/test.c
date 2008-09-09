#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <gen_param.h>
#include <gen_param_config.h>
#include <util.h>

int main(void) {
#define N 3
  gen_param_config_type * config = gen_param_config_alloc("/tmp/genp%d" , NULL);
  gen_param_type ** ens;
  ens = util_malloc(sizeof * ens * N , __func__);
  int i;

  for (i = 0; i < N; i++)
    ens[i] = gen_param_alloc( config );

  for (i = 0; i < N; i++)
    gen_param_initialize( ens[i] , i);

  for (i = 0; i < N; i++) {
    char * filename = util_alloc_sprintf("/tmp/GP%d" , i);
    gen_param_ecl_write( ens[i] , filename);
    free(filename);
  }

  for (i = 0; i < N; i++)
    gen_param_free( ens[i] );
  
  

  free(ens);
  

  
  gen_param_config_free( config );
}



