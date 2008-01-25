#ifndef __MEAS_MATRIX_H__
#define __MEAS_MATRIX_H__


typedef struct meas_matrix_struct meas_matrix_type;

meas_matrix_type * meas_matrix_alloc( int );
void               meas_matrix_free(meas_matrix_type * );
void               meas_matrix_add(meas_matrix_type * , int , double );
void               meas_vector_allocS(const meas_matrix_type * , double **);
#endif
