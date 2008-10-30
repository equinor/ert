#ifndef __SCALAR_CONFIG_H__
#define __SCALAR_CONFIG_H__
#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdbool.h>
#include <enkf_util.h>
#include <enkf_macros.h>
#include <void_arg.h>

typedef struct scalar_config_struct scalar_config_type;




struct scalar_config_struct {
  int data_size;            
  int internal_offset;
  double 	   * mean;
  double 	   * std;
  bool   	   * active;
  transform_ftype ** output_transform;
  char            ** output_transform_name;
  void_arg_type   ** void_arg;        
};

scalar_config_type *  scalar_config_alloc_empty(int);
void                  scalar_config_free(scalar_config_type *);
const          char * scalar_config_get_ensfile_ref(const scalar_config_type * );
const          char * scalar_config_get_eclfile_ref(const scalar_config_type * );
void                  scalar_config_transform(const scalar_config_type * , const double * , double *);
double                scalar_config_transform_item(const scalar_config_type *, double, int);
void                  scalar_config_truncate(const scalar_config_type * config , double *);
void                  scalar_config_fscanf_line(scalar_config_type * , int , FILE * );


CONFIG_SET_ECLFILE_HEADER(scalar);
CONFIG_SET_ENSFILE_HEADER(scalar);
GET_DATA_SIZE_HEADER(scalar);
CONFIG_SET_ECLFILE_HEADER_VOID(scalar);
CONFIG_SET_ENSFILE_HEADER_VOID(scalar);
VOID_FREE_HEADER(scalar_config);
#ifdef __cplusplus
}
#endif
#endif
