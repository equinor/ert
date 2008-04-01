#ifndef __MEAS_MATRIX_H__
#define __MEAS_MATRIX_H__
#include <meas_vector.h>

typedef struct meas_matrix_struct meas_matrix_type;

void               printf_matrix(const double *, int , int , int , int , const char * , const char * );
meas_vector_type * meas_matrix_iget_vector(const meas_matrix_type *, int );
meas_matrix_type * meas_matrix_alloc( int );
void               meas_matrix_free(meas_matrix_type * );
void               meas_matrix_add(meas_matrix_type * , int , double );
double           * meas_matrix_allocS(const meas_matrix_type * , int , int);
#endif
