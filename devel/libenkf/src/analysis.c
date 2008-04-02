/*
  Fortran routine - in libanalysis
*/

/*
subroutine enkfX5(X5, R, E, S, D, innov, nrens, nrobs, verbose, truncation,mode,update_randrot,istep,xpath)
! Computes the analysed ensemble for A using the EnKF or square root schemes.

   use mod_anafunc
   use m_multa
   implicit none

   integer, intent(in) :: istep
   character(Len=*), intent(in) :: xpath


   integer, intent(in) :: nrens            ! number of ensemble members
   integer, intent(in) :: nrobs            ! number of observations

   
   real, intent(inout) :: X5(nrens,nrens)  ! ensemble update matrix
   real, intent(in)    :: R(nrobs,nrobs)   ! matrix holding R (only used if mode=?1 or ?2)
   real, intent(in)    :: D(nrobs,nrens)   ! matrix holding perturbed measurments
   real, intent(in)    :: E(nrobs,nrens)   ! matrix holding perturbations (only used if mode=?3)
   real, intent(in)    :: S(nrobs,nrens)   ! matrix holding HA` 
   real, intent(in)    :: innov(nrobs)     ! vector holding d-H*mean(A)

   logical, intent(in) :: verbose          ! Printing some diagnostic output

   real, intent(in)    :: truncation       ! The ratio of variaince retained in pseudo inversion (0.99)

   integer, intent(in) :: mode             ! first integer means (EnKF=1, SQRT=2)
                                           ! Second integer is pseudo inversion
                                           !  1=eigen value pseudo inversion of SS'+(N-1)R
                                           !  2=SVD subspace pseudo inversion of SS'+(N-1)R
                                           !  3=SVD subspace pseudo inversion of SS'+EE'

   logical, intent(in) :: update_randrot   ! Normally true; false for all but first grid point
                                           ! updates when using local analysis since all grid
                                           ! points need to use the same rotation.

*/

#include <stdlib.h>
#include <util.h>
#include <obs_data.h>
#include <meas_matrix.h>

void analysis_set_stride(int ens_size , int nrobs , int * ens_stride , int * obs_stride) {
  *ens_stride = nrobs;
  *obs_stride = 1;
}



/**
   The actual routine m_enkfx5_mp_enkfx5_() is written in fortran, and
   found in the library libanalysis.
*/
void m_enkfx5_mp_enkfx5_(double * X , const double *R , const double * E , const double * S , const double * D , const double * innov , const int * nrens , 
			 const int * nrobs , const int * verbose , const double * truncation , const int * mode , const int * update_randrot , const int * istep , const char * xpath);



/**  

    Number of ensemble members -------------------->


    Number of observations
    |
    |
   \|/
     



     --------------------
    |S1  S4              |
S = |S2  S5              | 
    |S3  S6              |
     --------------------

     --------------------
    |E1  E4              |
E = |E1  E5              | 
    |E3  E6              |
     --------------------

     --------------------
    |D1  D4              |
D = |D2  D5              | 
    |D3  D6              |
     --------------------

     ------- 
    |R1  R4 |   This matrix *is* quadratic ...
R = |R2  R5 |
    |R3  R6 |
     -------


     --------------------
    |X1  X10             |
    |X2  X11             | 
    |X3  X12             |
    |X4  X13             | 
X = |X5  X14             | 
    |X6  X15             |
    |X7  X16             |  
    |X8  X17             |
    |X9  X18             |
     --------------------


     The C part of the code is written with strides, and (should) work
     for any low-level layout of the matrices, however the Fortran
     code insists on Fortran layout, i.e. the column index running
     fastest (i.e. with stride 1), as the numbering scheme above
     indicates.

  
     For the matrices S, E and D - it makes sense to talk of an
     ensemble direction (horizontally), and and observation direction
     (vertically). Considering element S1:

     * To go to next observation for the same ensemble member you go
       down to S2; in memory the distance between these two are 1,
       i.e. the obs_stride equals one. 

     * To go to the same next ensemble member, for the same
       observation you move horizontally to S4, in memory these are 3
       elements apart, i.e. the ens_stride equals 3.

     
     For the matrices R and X it does not make the same sense to talk
     of an ensemble direction and an observation direction as R has
     dimensisons nrobs x nrobs and X has dimensions nrens x nrens.
*/

double * analysis_allocX(int ens_size , int nrobs , const meas_matrix_type * meas_matrix, obs_data_type * obs_data , bool verbose , bool update_randrot) {
  int  update_randrot_int, verbose_int;
  const char * xpath 	  = NULL;
  const double alpha 	  = 1.50;
  const double truncation = 0.99;
  double *X , *R , *E , *S , *D , *innov;
  int mode , istep , ens_stride , obs_stride , iens, iobs;
  bool returnE; 
  
  istep      = -1;
  mode       = 22;
  returnE    = false;
  
  analysis_set_stride(ens_size , nrobs , &ens_stride , &obs_stride);
  X 	= util_malloc(ens_size * ens_size * sizeof * X, __func__);
  /*
    Must exclude both outliers and observations with zero ensemble
    variation *before* the matrices are allocated.
  */
  S 	= meas_matrix_allocS(meas_matrix , ens_stride , obs_stride);
  innov = obs_data_alloc_innov(obs_data , ens_size , ens_stride , obs_stride , S);
  R 	= obs_data_allocR(obs_data , ens_size , ens_stride , obs_stride , innov , S , alpha);
  D 	= obs_data_allocD(obs_data , ens_size , ens_stride , obs_stride , S , returnE , &E);
  obs_data_scale(obs_data , ens_size  , ens_stride , obs_stride, S , E , D , R , innov);
  

  /*
     Substracting mean value of S
  */
  for (iobs = 0; iobs < nrobs; iobs++) {
    double S1 = 0;
    for (iens = 0; iens < ens_size; iens++) {
      int index = iobs * obs_stride + iens * ens_stride;
      S1 += S[index];
    }
    S1 = S1 / ens_size;
    for (iens = 0; iens < ens_size; iens++) {
      int index = iobs * obs_stride + iens * ens_stride;
      S[index] -= S1;
    }
  } 
  
  verbose_int        = util_C2f90_bool(verbose);
  update_randrot_int = util_C2f90_bool(update_randrot);
  

  if (verbose) {
    printf_matrix(R , nrobs , nrobs    , 1 , nrobs , "R" , " %8.3lg ");
    printf("\n");
    
    printf_matrix(D , nrobs , ens_size , obs_stride , ens_stride , "D" , " %8.3lg ");
    printf("\n");
    
    printf_matrix(S , nrobs , ens_size , obs_stride , ens_stride , "S" , " %8.3lg ");
    printf("\n");

    printf_matrix(innov , nrobs , 1 , 1 , 1 , "Innov" , " %8.3lg ");
  }

  m_enkfx5_mp_enkfx5_(X , 
		      R , 
		      E , 
		      S , 
		      D , 
		      innov , 
		      (const int *) &ens_size        	, 
		      (const int *) &nrobs        	, 
		      (const int *) &verbose_int  	, 
		      (const double *) &truncation 	, 
		      (const int *) &mode         	,     
		      (const int *) &update_randrot_int , 
		      (const int *) &istep              , 
		      xpath);
  
  free(S);
  free(R);
  free(D);
  free(innov);
  if (E != NULL) free(E);

  printf_matrix(X , ens_size , ens_size , 1 , ens_size , "X" , " %8.3lg" );
  return X;
}


/*

  initX(X , R , E , S , D , innov , ....);

*/
  
