#ifndef __MEAS_MATRIX_H__
#define __MEAS_MATRIX_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <meas_vector.h>
#include <stdbool.h>

typedef struct meas_matrix_struct meas_matrix_type;

void               meas_matrix_reset(meas_matrix_type * );
void               fprintf_matrix(FILE * stream, const double *, int , int , int , int , const char * , const char * );
void               fwrite_matrix(const char *, const double *, int , int , int , int , const char * , const char * );
void               printf_matrix(const double *, int , int , int , int , const char * , const char * );
meas_vector_type * meas_matrix_iget_vector(const meas_matrix_type *, int );
meas_matrix_type * meas_matrix_alloc( int );
void               meas_matrix_free(meas_matrix_type * );
void               meas_matrix_add(meas_matrix_type * , int , double );
double           * meas_matrix_allocS(const meas_matrix_type * , int , int , int , double ** , const bool *);
void               meas_matrix_allocS_stats(const meas_matrix_type * , double **, double **);
#ifdef __cplusplus
}
#endif
#endif
