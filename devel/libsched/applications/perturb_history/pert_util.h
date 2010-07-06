#ifndef __PERT_UTIL_H__
#define __PERT_UTIL_H__
#include <double_vector.h>
#include <bool_vector.h>
#include <time_t_vector.h>

void                 rand_dbl(int N , double max , double *R);
double               rand_normal(double mean , double std);
void                 rand_stdnormal_vector(int size , double *R);
void                 fscanf_2ts(const time_t_vector_type * time_vector , const char * filename , double_vector_type * ts1 , double_vector_type * ts2, bool_vector_type * pts);


#endif
