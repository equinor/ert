#ifndef __ENKF_ANALYSIS_H__
#define __ENKF_ANALYSIS_H__

#include <matrix.h>



void       enkf_analysis_standard_lowrankCinv(matrix_type * X5 , matrix_type * R , const matrix_type * S , const matrix_type * D , double truncation);


#endif
