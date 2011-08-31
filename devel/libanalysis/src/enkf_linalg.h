#ifndef __ENKF_LINALG_H__
#define __ENKF_LINALG_H__

#include <matrix_lapack.h>

void enkf_linalg_svdS(const matrix_type * S , 
                      double truncation , 
                      int ncomp ,
                      dgesvd_vector_enum jobVT , 
                      double * sig0, 
                      matrix_type * U0 , 
                      matrix_type * V0T);



void enkf_linalg_lowrankCinv__(const matrix_type * S , 
                               const matrix_type * R , 
                               matrix_type * V0T , 
                               matrix_type * Z, 
                               double * eig , 
                               matrix_type * U0, 
                               double truncation, 
                               int ncomp);



void enkf_linalg_lowrankCinv(const matrix_type * S , 
                             const matrix_type * R , 
                             matrix_type * W       , /* Corresponding to X1 from Eq. 14.29 */
                             double * eig          , /* Corresponding to 1 / (1 + Lambda_1) (14.29) */
                             double truncation     ,
                             int    ncomp);



void enkf_linalg_genX2(matrix_type * X2 , const matrix_type * S , const matrix_type * W , const double * eig);
void enkf_linalg_genX3(matrix_type * X3 , const matrix_type * W , const matrix_type * D , const double * eig);


#endif
