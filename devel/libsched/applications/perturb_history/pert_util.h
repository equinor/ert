#ifndef __PERT_UTIL_H__
#define __PERT_UTIL_H__



typedef enum { OIL   = 0,
               WATER = 1,
               GAS   = 2} phase_type;


phase_type           phase_from_string(const char * __phase_string); 
const char         * get_phase_name( phase_type phase);
void                 rand_dbl(int N , double max , double *R);
double               rand_normal(double mean , double std);
void                 rand_stdnormal_vector(int size , double *R);
void                 fscanf_2ts(const time_t_vector_type * time_vector , const char * filename , double_vector_type * ts1 , double_vector_type * ts2);


#endif
