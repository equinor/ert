#ifndef __EQUIL_CONFIG_H__
#define __EQUIL_CONFIG_H__

#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <enkf_config.h>
#include <scalar_config.h>

typedef struct equil_config_struct equil_config_type;

struct equil_config_struct {
  CONFIG_STD_FIELDS;

  double * datum_depth;
  double * datum_P;
  double * oil_water_Pc;
  double * gas_oil_Pc;
  int    * live_oil_init_mode;
  int    * black_oil_wgas_init_mode;
  int    * init_accuracy;

  scalar_config_type * scalar_config;
};

void                  equil_config_ecl_write(const equil_config_type *   , const double *  , const double * , FILE * );
int                   equil_config_get_nequil(const equil_config_type * );
equil_config_type   * equil_config_alloc(int);
void                  equil_config_free(equil_config_type *);
int                   equil_config_get_nequil(const equil_config_type *);
equil_config_type   * equil_config_fscanf_alloc(const char * );

VOID_FUNC_HEADER(equil_config_free);

#endif
