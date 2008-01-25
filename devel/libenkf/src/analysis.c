#include <stdlib.h>
#include <util.h>

/*
  Fortran routine - linked in dirctly with the object file 
*/

/*
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



void analysis_set_stride(int ens_size , int nrobs , int * ens_stride , int * obs_stride) {
  *ens_stride = nrobs;
  *obs_stride = 1;
}




void analysis_initS(void) {

}


void m_enkfx5_mp_enkfx5_(double * X , const double *R , const double * E , const double * S , const double * D , const double * innov , const int * nrens , 
			 const int * nrobs , const int * verbose , const double * truncation , const int * mode , const int * update_randrot , const int * istep , const char * xpath);



void initX(int nrens , int nrobs , bool verbose , bool update_randrot) {
  int  update_randrot_int, verbose_int;
  double *X , *R , *E , *S , *D , *innov;
  int  truncation , mode , istep;
  char * xpath;

  verbose_int        = util_C2f90_bool(verbose);
  update_randrot_int = util_C2f90_bool(update_randrot);
  
  m_enkfx5_mp_enkfx5_(X , 
		      R , 
		      E , 
		      S , 
		      D , 
		      innov , 
		      (const int *) &nrens        	, 
		      (const int *) &nrobs        	, 
		      (const int *) &verbose_int  	, 
		      (const double *) truncation 	, 
		      (const int *) &mode         	,     
		      (const int *) &update_randrot_int , 
		      (const int *) &istep              , 
		      xpath);
  
}
