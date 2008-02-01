#ifndef __ANALYSIS_H__
#define __ANALYSIS_H__



void     analysis_set_stride(int , int , int * , int * );
double * analysis_allocX(int , int , const meas_matrix_type * , const obs_data_type * , bool , bool );

#endif
