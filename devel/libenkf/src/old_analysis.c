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

#include <string.h>
#include <stdlib.h>
#include <util.h>
#include <obs_data.h>
#include <meas_matrix.h>
#include <enkf_types.h>
#include <math.h>
#include <old_analysis.h>
#include <config.h>
#include <matrix.h>


struct analysis_config_struct {
  bool                   merge_observations;  /* When observing from time1 to time2 - should ALL observations in between be used? */
  double 	         truncation;
  double 	         overlap_alpha;
  enkf_mode_type         enkf_mode;
  pseudo_inversion_type  inversion_mode;
  int                    fortran_enkf_mode; 
};



static analysis_config_type * analysis_config_alloc__(double truncation , double overlap_alpha , enkf_mode_type enkf_mode , bool merge_observations) {
  analysis_config_type * config = util_malloc( sizeof * config , __func__);

  config->merge_observations = merge_observations;
  config->truncation         = truncation;
  config->overlap_alpha      = overlap_alpha;
  config->enkf_mode          = enkf_mode;
  config->inversion_mode     = SVD_SS_N1_R;
  
  config->fortran_enkf_mode  = config->enkf_mode + config->inversion_mode;
  return config;
}


analysis_config_type * analysis_config_alloc(const config_type * config) {
  double truncation 	    = strtod( config_get(config , "ENKF_TRUNCATION") , NULL);
  double alpha      	    = strtod( config_get(config , "ENKF_ALPHA") , NULL);
  bool   merge_observations = config_get_as_bool(config , "ENKF_MERGE_OBSERVATIONS");
  const char * enkf_mode_string = config_get(config , "ENKF_MODE");
  enkf_mode_type enkf_mode = ENKF_SQRT; /* Compiler shut up */

  if (strcmp(enkf_mode_string,"STANDARD") == 0)
    enkf_mode = ENKF_STANDARD;
  else if (strcmp(enkf_mode_string , "SQRT") == 0)
    enkf_mode = ENKF_SQRT;
  else
    util_abort("%s: internal error : enkf_mode:%s not recognized \n",__func__ , enkf_mode_string);

  return analysis_config_alloc__(truncation , alpha , enkf_mode , merge_observations);
}


bool analysis_config_merge_observations(const analysis_config_type * config) {
  return config->merge_observations;
}


void analysis_config_free(analysis_config_type * config) {
  free(config);
}


static int __C2f90_bool(bool c_bool) {
  /* This is true for the ifort compiler ... */
  if (c_bool)
    return -1;
  else
    return 0;
}  


/*****************************************************************/

void analysis_X5(matrix_type * X      , 
		 const matrix_type * R,
		 const matrix_type * E,
		 const matrix_type * S,
		 const matrix_type * D, 
		 const matrix_type * innov) {
  
  const int ens_size     = matrix_get_columns( S );
  const int nrobs_active = matrix_get_rows( S );
  
}



void analysis_deactivate_outliers(obs_data_type * obs_data , meas_matrix_type * meas_matrix) {

}

   



/*****************************************************************/


void analysis_set_stride(int ens_size , int nrobs , int * ens_stride , int * obs_stride) {
  *ens_stride = nrobs;
  *obs_stride = 1;
}



/**
   The actual routine __m_enkfx5__enkfx5() is written in fortran, and
   found in the library libanalysis. Observe that the fortran compiler
   mangles the names, so with new fortran compiler we will get a new
   name here ...
*/
void __m_enkfx5__enkfx5(double * X , const double *R , const double * E , const double * S , const double * D , const double * innov , const int * nrens , 
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

     --------------------   This matrix is only used for mode = ?3 => 
    |E1  E4              |  SVD subspace pseudo inversion of SS'+EE'. 
E = |E1  E5              | 
    |E3  E6              |
     --------------------

     --------------------    This matrix is only used for the EnKF 
    |D1  D4              |   schemes with perturbed measurements.  
D = |D2  D5              | 
    |D3  D6              |
     --------------------

     ------- 
    |R1  R4 |                 This matrix *is* quadratic ...
R = |R2  R5 |
    |R3  R6 |
     -------


     --------------------     This matrix *is* quadratic ...
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

     * To go to the next ensemble member, for the same observation, you
       move horizontally to S4, in memory these are 3 elements apart,
       i.e. the ens_stride equals 3.

     
     For the matrices R and X it does not make the same sense to talk
     of an ensemble direction and an observation direction as R has
     dimensisons nrobs x nrobs and X has dimensions nrens x nrens.
*/


double * old_analysis_allocX(int ens_size , int nrobs_total , const meas_matrix_type * meas_matrix, obs_data_type * obs_data , bool verbose , bool update_randrot , const analysis_config_type * config) {
  int  update_randrot_int, verbose_int;
  const char * xpath 	  = NULL;
  const double alpha 	  = config->overlap_alpha;
  const double truncation = config->truncation;
  const int    mode       = config->fortran_enkf_mode;  
  bool  * active_obs;
  double *X ;
  int istep , ens_stride , obs_stride , nrobs_active;
  
  istep      = -1;     /* Must be <= 0 to avoid writing on xpath - which will fail. */

  
  /*
    
    mode     Need E     Need D
    --------------------------- 
    11       tmp        Yes 
    12       tmp        Yes
    13       Yes        Yes
    21       No         No
    22       No         No
    23       Yes        No
    --------------------------
    
  */


  /*
    Must exclude both outliers and observations with zero ensemble
    variation *before* the matrices are allocated.
  */

  X = NULL;
  {
    /*
      This code block is used deactivate measurements with:

       1. Zero ensemble variation
       2. Too large deviation between ensemble and observation

      The variables _meanS, _stdS and _innov are computed for *ALL*
      observations, whereas the variables meanS, and innov further
      down only contain the active observations.
    */
    
    double *_meanS , *_stdS , *_innov; 
    
    meas_matrix_allocS_stats(meas_matrix , &_meanS , &_stdS);
    _innov = obs_data_alloc_innov(obs_data , _meanS);
    obs_data_deactivate_outliers(obs_data , _innov , _stdS , 1e-6 , alpha , &nrobs_active , &active_obs);
    obs_data_fprintf(obs_data , stdout , _meanS , _stdS);

    free(_innov);
    free(_meanS);
    free(_stdS);
  }

  /**
     
     0. Deaktiver outliers.
     1. Alloker S og <S> fra obs_data / meas_matrix   
     2. Alloker R fra obs_data.
     3. Allokerer E fra obs_data.
     4. Alloker D fra obs_data , E, S og <S>. (Denne trenger egentlig selve S)
     5. Alloker innov fra obs_data og <S>.   Etter punkt 5 kan man slippe <S>.
     6. Skaler S,D,E,R,innov fra obs_data.
     7. Beregn X(S,D,E,R,innov);
  
  */


  if (nrobs_active > 0) {
    double  * innov = NULL;
    double  * meanS = NULL;
    double  *R , *E , *S , *D;
    
    analysis_set_stride(ens_size , nrobs_active , &ens_stride , &obs_stride);
    S = meas_matrix_allocS(meas_matrix , nrobs_active , ens_stride , obs_stride , &meanS , active_obs);
    
    /* Just a check of the matrix based routines */
    {
      matrix_type * __S;
      __S   = meas_matrix_allocS__(meas_matrix , nrobs_active , NULL /*&meanS*/ , active_obs);
      
      if (memcmp( S , matrix_get_data(__S) , nrobs_active * ens_size * sizeof * S) != 0)
	util_abort("%s: matrixS error \n");
      
      matrix_free( __S );
    }
    
    R 	  = obs_data_allocR(obs_data);
    E     = obs_data_allocE(obs_data , ens_size , ens_stride , obs_stride);
    D     = obs_data_allocD(obs_data , ens_size , ens_stride , obs_stride , E , S , meanS);
    innov = obs_data_alloc_innov(obs_data , meanS);

    obs_data_scale(obs_data , ens_size  , ens_stride , obs_stride, S , E , D , R , innov);
    X 	= util_malloc(ens_size * ens_size * sizeof * X, __func__);

  
    verbose_int        = __C2f90_bool(verbose);
    update_randrot_int = __C2f90_bool(update_randrot);
    
    if (verbose) {
      printf_matrix(R , nrobs_active , nrobs_active    , 1 , nrobs_active , "R" , " %10.6f ");
      fwrite_matrix("enkf_debug_dump_R.txt", R , nrobs_active , nrobs_active    , 1 , nrobs_active , "R" , " %10.6f ");
      printf("\n");
      
      printf_matrix(D , nrobs_active , ens_size , obs_stride , ens_stride , "D" , " %10.6f ");
      fwrite_matrix("enkf_debug_dump_D.txt", D , nrobs_active , ens_size , obs_stride , ens_stride , "D" , " %10.6f ");
      printf("\n");
    
      printf_matrix(S , nrobs_active , ens_size , obs_stride , ens_stride , "S" , " %10.6f ");
      fwrite_matrix("enkf_debug_dump_S.txt" , S , nrobs_active , ens_size , obs_stride , ens_stride , "S" , " %10.6f ");
      printf("\n");

      printf_matrix(innov , nrobs_active , 1 , 1 , 1 , "Innov" , " %10.6f ");
      fwrite_matrix("enkf_debug_dump_innov.txt", innov , nrobs_active , 1 , 1 , 1 , "Innov" , " %10.6f ");
      printf("\n");
    }
    
    __m_enkfx5__enkfx5(X , 
			R , 
			E , 
			S , 
			D , 
		       innov , 
		       (const int *) &ens_size        	, 
		       (const int *) &nrobs_active     , 
		       (const int *) &verbose_int  	, 
		       (const double *) &truncation 	, 
		       (const int *) &mode         	,     
		       (const int *) &update_randrot_int , 
		       (const int *) &istep              , 
		       xpath);

    /*
    {
      matrix_type * __S = meas_matrix_allocS__(meas_matrix , nrobs_active , NULL , active_obs);
      matrix_type * __R = obs_data_allocR__( obs_data );
      matrix_type * __E = obs_data_allocE__( obs_data , ens_size );
      matrix_type * __D = obs_data_allocD__( obs_data , __E , __S , meanS);
      matrix_type * __X5 = matrix_alloc(ens_size , ens_size);
      
      obs_data_scale__(obs_data , __S , __E , __D , __R , innov);
      
      enkf_analysis_standard_lowrankCinv(__X5 , __R , __S , __D , truncation );
      printf_matrix(X , ens_size , ens_size , 1 , ens_size , "X" , " %10.6f" );
      matrix_pretty_print(__X5 , "X5" , " %10.6f");
      
      //if (memcmp( X  , matrix_get_data( __X5) , ens_size * ens_size * sizeof * X) != 0)
      //util_abort("%s: X5 error \n",__func__);
	
      matrix_free( __X5);
      matrix_free( __S );
      matrix_free( __R );
      matrix_free( __E );
      matrix_free( __D );
      }
    */
    
    if (verbose) {
      printf_matrix(X , ens_size , ens_size , 1 , ens_size , "X" , " %10.6f" );
      fwrite_matrix("enkf_debug_dump_X.txt", X , ens_size , ens_size , 1 , ens_size , "X" , " %10.6f" );
    }

    {
      int col;
      for (col = 0; col < ens_size; col ++) {
	double col_sum = 0;
	int row;
	for (row = 0; row < ens_size; row++) {
	  int index = row + col * ens_size;
	  col_sum += X[index];
	}
	if (fabs(col_sum - 1.0) > 0.0001) 
	  util_abort("%s: something is seriously broken. col:%d  col_sum = %g != 1.0 - ABORTING\n",__func__ , col , col_sum);
      }
    }
    
      
    free(E);
    free(innov);
    free(D);  
    free(R);
    free(meanS);
    free(S);
    

  } else 
    printf("** No active observations ** \n");

  return X;
}


/*

  initX(X , R , E , S , D , innov , ....);

*/
  
