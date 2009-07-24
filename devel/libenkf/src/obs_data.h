#ifndef __OBS_DATA_H__
#define __OBS_DATA_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <stdio.h>
#include <stdbool.h>
#include <matrix.h>
#include <meas_matrix.h>

typedef struct obs_data_struct      obs_data_type;
typedef struct obs_data_node_struct obs_data_node_type;

const char *  obs_data_node_get_keyword( const obs_data_node_type * node );
double 	      obs_data_node_get_std( const obs_data_node_type * node );
double 	      obs_data_node_get_value( const obs_data_node_type * node );
bool          obs_data_node_active(const obs_data_node_type * node);


obs_data_type      * obs_data_alloc();
obs_data_node_type * obs_data_iget_node( const obs_data_type * obs_data , int index );
void          	     obs_data_free(obs_data_type *);
void          	     obs_data_add(obs_data_type * , double , double , const char * );
void          	     obs_data_reset(obs_data_type * obs_data);
double        	   * obs_data_allocD(const obs_data_type * obs_data , int ens_size, int ens_stride , int obs_stride , const double * E , const double * S , const double * meanS);
matrix_type   	   * obs_data_allocD__(const obs_data_type * obs_data , const matrix_type * E  , const matrix_type * S);
double        	   * obs_data_allocR(obs_data_type *);
matrix_type   	   * obs_data_allocR__(obs_data_type * obs_data);
double        	   * obs_data_alloc_innov(const obs_data_type * , const double *);
double             * obs_data_alloc_innov__(const obs_data_type * obs_data , const meas_matrix_type * meas_matrix);
double        	   * obs_data_allocE(const obs_data_type * obs_data , int ens_size, int ens_stride, int obs_stride);
matrix_type   	   * obs_data_allocE__(const obs_data_type * obs_data , int ens_size);
void          	     obs_data_scale(const obs_data_type * , int , int , int , double * , double *, double *, double * , double *);
void                 obs_data_scale__(const obs_data_type * obs_data , matrix_type *S , matrix_type *E , matrix_type *D , matrix_type *R , double *innov);
int           	     obs_data_get_nrobs(const obs_data_type * );
void          	     obs_data_deactivate_outliers(obs_data_type * , const double * , const double *, double , double , int * , bool **);
void          	     obs_data_fprintf(const obs_data_type * , FILE * , const double * , const double *);
void          	     obs_data_iget_value_std(const obs_data_type * obs_data , int index , double * value ,  double * std);
void          	     obs_data_deactivate_obs(obs_data_type * obs_data , int iobs,const char * msg);
int                  obs_data_get_active_size( const obs_data_type  * obs_data );

#ifdef __cplusplus
}
#endif
#endif
