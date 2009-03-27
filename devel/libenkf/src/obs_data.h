#ifndef __OBS_DATA_H__
#define __OBS_DATA_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stdbool.h>

typedef struct obs_data_struct obs_data_type;

obs_data_type * obs_data_alloc();
void            obs_data_free(obs_data_type *);
void            obs_data_add(obs_data_type * , double , double , const char * );
void            obs_data_reset(obs_data_type * obs_data);
double        * obs_data_allocD(const obs_data_type * , int , int , int , const double * , const double * , bool , double ** );
double        * obs_data_allocR(obs_data_type *);
double        * obs_data_alloc_innov(const obs_data_type * , const double *);
void            obs_data_scale(const obs_data_type * , int , int , int , double * , double *, double *, double * , double *);
int             obs_data_get_nrobs(const obs_data_type * );
void            obs_data_deactivate_outliers(obs_data_type * , const double * , const double *, double , double , int * , bool **);
void            obs_data_fprintf(const obs_data_type * , FILE * , const double * , const double *);

#ifdef __cplusplus
}
#endif
#endif
