#ifndef __ENKF_LINALG_H__
#define __ENKF_LINALG_H__

#include <matrix_lapack.h>
#include <matrix.h>


void enkf_linalg_get_PC( const matrix_type * S0, 
                         const matrix_type * dObs , 
                         double truncation,
                         int ncomp, 
                         matrix_type * PC,
                         matrix_type * PC_obs );


void enkf_linalg_init_stdX( matrix_type * X , 
                            const matrix_type * S , 
                            const matrix_type * D , 
                            const matrix_type * W , 
                            const double * eig , 
                            bool bootstrap);


void enkf_linalg_init_sqrtX(matrix_type * X5      , 
                            const matrix_type * S , 
                            const matrix_type * randrot , 
                            const matrix_type * innov , 
                            const matrix_type * W , 
                            const double * eig , 
                            bool bootstrap);


void enkf_linalg_Cee(matrix_type * B, int nrens , const matrix_type * R , const matrix_type * U0 , const double * inv_sig0);

int enkf_linalg_svdS(const matrix_type * S , 
                     double truncation , 
                     int ncomp ,
                     dgesvd_vector_enum jobVT , 
                     double * sig0, 
                     matrix_type * U0 , 
                     matrix_type * V0T);



matrix_type * enkf_linalg_alloc_innov( const matrix_type * dObs , const matrix_type * S);

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

void enkf_linalg_meanX5(const matrix_type * S , 
                        const matrix_type * W , 
                        const double * eig    , 
                        const matrix_type * innov ,
                        matrix_type * X5);


void enkf_linalg_X5sqrt(matrix_type * X2 , matrix_type * X5 , const matrix_type * randrot, int nrobs);

matrix_type * enkf_linalg_alloc_mp_randrot(int ens_size , rng_type * rng);
void          enkf_linalg_set_randrot( matrix_type * Q  , rng_type * rng);
void          enkf_linalg_checkX(const matrix_type * X , bool bootstrap);
#endif
