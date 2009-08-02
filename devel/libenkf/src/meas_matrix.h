#ifndef __MEAS_MATRIX_H__
#define __MEAS_MATRIX_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <meas_vector.h>
#include <stdbool.h>
#include <matrix.h>

typedef struct meas_matrix_struct meas_matrix_type;

void               meas_matrix_reset(meas_matrix_type * );
void               fprintf_matrix(FILE * stream, const double *, int , int , int , int , const char * , const char * );
void               fwrite_matrix(const char *, const double *, int , int , int , int , const char * , const char * );
void               printf_matrix(const double *, int , int , int , int , const char * , const char * );
meas_vector_type * meas_matrix_iget_vector(const meas_matrix_type *, int );
meas_matrix_type * meas_matrix_alloc( int );
void               meas_matrix_free(meas_matrix_type * );
void               meas_matrix_add(meas_matrix_type * , int , double );
matrix_type      * meas_matrix_allocS__(const meas_matrix_type * matrix);
void               meas_matrix_deactivate(meas_matrix_type * meas_matrix, int index);
void               meas_matrix_calculate_ens_stats(meas_matrix_type * matrix);
double             meas_matrix_iget_ens_mean(const meas_matrix_type * matrix , int index);
double             meas_matrix_iget_ens_std(const meas_matrix_type * matrix , int index);
void               meas_matrix_iget_ens_mean_std( const meas_matrix_type * matrix , int index , double * mean , double * std);
int                meas_matrix_get_ens_size( const meas_matrix_type * meas_matrix );
#ifdef __cplusplus
}
#endif
#endif
