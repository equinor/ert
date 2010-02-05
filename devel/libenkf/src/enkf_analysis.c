#include <util.h>
#include <math.h>
#include <matrix.h>
#include <matrix_lapack.h>
#include <matrix_blas.h>
#include <meas_matrix.h>
#include <obs_data.h>
#include <analysis_config.h>
#include <enkf_util.h>

/**
   The static functions at the top are identical to the old fortran
   functions from mod_anafunc() with the same name.
*/


static void genX3(matrix_type * X3 , const matrix_type * W , const matrix_type * D , const double * eig) {
  const int nrobs = matrix_get_rows( D );
  const int nrens = matrix_get_columns( D );
  const int nrmin = util_int_min( nrobs , nrens );
  int i,j;
  matrix_type * X1 = matrix_alloc(nrmin , nrobs);
  matrix_type * X2 = matrix_alloc(nrmin , nrens);

  for (i=0; i < nrmin; i++)
    for (j=0; j < nrobs; j++)
      matrix_iset(X1 , i , j , eig[i] * matrix_iget(W , j , i));
  
  matrix_matmul(X2 , X1 , D);
  matrix_matmul(X3 , W  , X2);   
  
  matrix_free( X1 );
  matrix_free( X2 );
}



static void genX2(matrix_type * X2 , const matrix_type * S , const matrix_type * W , const double * eig) {
  const int nrens = matrix_get_columns( S );
  const int idim  = matrix_get_rows( X2 );
  matrix_dgemm(X2 , W , S , true , false , 1.0 , 0.0);
  { 
    int i,j;
    for (j=0; j < nrens; j++)
      for (i=0; i < idim; i++)
	matrix_imul(X2 , i,j , sqrt(eig[i]));
  }
}


/** 
    Mimicks the original fortran function with the same name. 
    
    Input is the S - matrix. U0 comes directly from dgesvd, sig0 has
    been seriously massaged.
*/


static void svdS(const matrix_type * S , matrix_type * U0 , double * sig0, double truncation) {
  int num_singular_values = util_int_min( matrix_get_rows( S ) , matrix_get_columns( S ));
  {
    /* 
       The svd routine will destroy the contents of the input matrix,
       we therefor have to store a copy of S before calling it. The
       fortran implementation seems to handle this automagically??
    */
    matrix_type * workS     = matrix_alloc_copy( S );
    matrix_dgesvd(DGESVD_MIN_RETURN , DGESVD_NONE , workS , sig0 , U0 , NULL);                   /* Have singular values in s0, and left hand singular vectors in U0 */
    matrix_free( workS );
  }
  
  {
    double total_sigma2    = 0;
    double running_sigma2  = 0;
    int    num_significant = 0;
    int i;
    for (i=0; i < num_singular_values; i++)
      total_sigma2 += sig0[i] * sig0[i];

    for (i=0; i < num_singular_values; i++) {
      if (running_sigma2 / total_sigma2 < truncation) {  /* Include one more singular value ? */
	num_significant++;
	running_sigma2 += sig0[i] * sig0[i];
      } else 
	break;
    }

    /* Explicitly setting the insignificant singular values to zero. */
    for (i=num_significant; i < num_singular_values; i++)
      sig0[i] = 0;                                     
    
    /* Inverting the significant singular values */
    for (i = 0; i < num_significant; i++)
      sig0[i] = 1.0 / sig0[i];
  }
}




static void lowrankCee(matrix_type * B, int nrens , const matrix_type * R , const matrix_type * U0 , const double * sig0) {
  const int nrmin = matrix_get_rows( B );
  {
    matrix_type * X0 = matrix_alloc( nrmin , matrix_get_rows( R ));
    matrix_dgemm(X0 , U0 , R  , true  , false , 1.0 , 0.0);  /* X0 = U0^T * R */
    matrix_dgemm(B  , X0 , U0 , false , false , 1.0 , 0.0);  /* B = X0 * U0 */
    matrix_free( X0 );
  }    
  
  {
    int i ,j;

    /* Funny code ?? */
    for (j=0; j < matrix_get_columns( B ) ; j++)
      for (i=0; i < matrix_get_rows( B ); i++)
	matrix_imul(B , i , j , sig0[i]);

    for (j=0; j < matrix_get_columns( B ) ; j++)
      for (i=0; i < matrix_get_rows( B ); i++)
	matrix_imul(B , i , j , sig0[j]);
  }
  
  matrix_scale(B , nrens - 1.0);
}



static void eigC(matrix_type * R , matrix_type * Z , double * eig_values) {
  matrix_dsyevx_all( DSYEVX_AUPPER , R , eig_values , Z);
}



static void lowrankCinv(const matrix_type * S , const matrix_type * R , matrix_type * W , double * eig , double truncation) {
  const int nrobs = matrix_get_rows( S );
  const int nrens = matrix_get_columns( S );
  const int nrmin = util_int_min( nrobs , nrens );
  
  matrix_type * B    = matrix_alloc( nrmin , nrmin );
  matrix_type * U0   = matrix_alloc( nrobs , nrmin );
  matrix_type * Z    = matrix_alloc( nrmin , nrmin );
  double * sig0      = util_malloc( nrmin * sizeof * sig0 , __func__);
  
  svdS(S , U0 , sig0 , truncation);             
  lowrankCee( B , nrens , R , U0 , sig0);       
 
  eigC( B , Z , eig );                          
  {
    int i,j;
    for (i=0; i < nrmin; i++) 
      eig[i] = 1.0 / (1 + eig[i]);

    for (j=0; j < nrmin; j++)
      for (i=0; i < nrmin; i++)
	matrix_imul(Z , i , j , sig0[i]);
  }
  matrix_matmul(W , U0 , Z);

  free( sig0 );
  matrix_free( U0 );
  matrix_free( B  );
  matrix_free( Z  );
}




static void meanX5(const matrix_type * S , const matrix_type * W , const double * eig , const double * innov , matrix_type * X5) {
  const int nrens = matrix_get_columns( S );
  const int nrobs = matrix_get_rows( S );
  const int nrmin = util_int_min( nrobs , nrens );
  double * work   = util_malloc( (2 * nrmin + nrobs + nrens) * sizeof * work , __func__);
  {
    double * y1 = &work[0];
    double * y2 = &work[nrmin];
    double * y3 = &work[2*nrmin];
    double * y4 = &work[2*nrmin + nrobs]; 
    
    if (nrobs == 1) {
      /* Is this special casing necessary ??? */
      y1[0] = matrix_iget(W , 0,0) * innov[0];
      y2[0] = eig[0] * y1[0];
      y3[0] = matrix_iget(W , 0, 0) *y2[0];
      for (int iens = 0; iens < nrens; iens++)
        y4[iens] = y3[0] * matrix_iget(S , 0, iens);
    } else {
      matrix_dgemv(W , innov , y1 , true , 1.0, 0.0);   /* y1 = Trans(W) * innov */
      for (int i= 0; i < nrmin; i++)
        y2[i] = eig[i] * y1[i];                         /* y2 = eig * y1      */
      matrix_dgemv(W , y2 , y3 , false , 1.0 , 0.0);    /* y3 = W * y2;       */ 
      matrix_dgemv(S , y3 , y4 , true  , 1.0 , 0.0);    /* y4 = Trans(S) * y3 */
    }
    
    for (int iens = 0; iens < nrens; iens++)
      matrix_set_column(X5 , y4 , iens );
    
    matrix_shift(X5 , 1.0/nrens);
  }
  free( work );
}



/**
   The original fortran code has a mode input flag, and intent(inout)
   on the nrmin variable. It looks completely broken.

   In the fortran code X2 is intent(in) - but that must be handled by
   some magic, because the dgesvd() routine does write on X2; however
   that seems to be OK.
*/  

static void X5sqrt(matrix_type * X2 , matrix_type * X5 , const matrix_type * randrot, int nrobs) { 
  const int nrens   = matrix_get_columns( X5 );
  const int nrmin   = util_int_min( nrobs , nrens );
  matrix_type * VT  = matrix_alloc( nrens , nrens );
  double * sig      = util_malloc( nrmin * sizeof * sig , __func__);
  double * isig     = util_malloc( nrmin * sizeof * sig , __func__);

  matrix_dgesvd(DGESVD_NONE , DGESVD_ALL , X2 , sig , NULL , VT);
  {
    matrix_type * X3   = matrix_alloc( nrens , nrens );
    matrix_type * X33  = matrix_alloc( nrens , nrens );
    matrix_type * X4   = matrix_alloc( nrens , nrens );
    matrix_type * IenN = matrix_alloc( nrens , nrens );
    int i,j;
    for (i = 0; i < nrmin; i++)
      isig[i] = sqrt( util_double_max( 1.0 - sig[i]*sig[i]  ,0.0));
    
    for (j = 0; j < nrens; j++)
      for (i = 0; i < nrens; i++)
        matrix_iset(X3 , i , j , matrix_iget(VT , j , i));
    
    for (j=0; j< nrmin; j++)
      matrix_scale_column(X3 , j , isig[j]);
    
    matrix_dgemm(X33 , X3 , VT , false , false , 1.0 , 0.0);        /* X33 = X3   * VT */
    if (randrot != NULL)
      matrix_dgemm(X4  , X33 , randrot , false, false , 1.0 , 0.0);   /* X4  = X33  * Randrot */             
    else
      matrix_assign(X4 , X33);
    
    matrix_set(IenN , -1.0/ nrens);
    for (i = 0; i < nrens; i++)
      matrix_iadd(IenN , i , i , 1.0);
    
    matrix_dgemm(X5  , IenN , X4 , false , false , 1.0 , 1.0);      /* X5  = IenN * X4 + X5 */

    matrix_free( X3   );
    matrix_free( X33  );
    matrix_free( X4   );
    matrix_free( IenN );
  }

  free(sig);
  free(isig);
  matrix_free( VT );
}





/**
    This routine generates a real orthogonal random matrix.
    The algorithm is the one by
       Francesco Mezzadri (2007), How to generate random matrices from the classical
       compact groups, Notices of the AMS, Vol. 54, pp 592-604.
    1. First a matrix with independent random normal numbers are simulated.
    2. Then the QR decomposition is computed, and Q will then be a random orthogonal matrix.
    3. The diagonal elements of R are extracted and we construct the diagonal matrix X(j,j)=R(j,j)/|R(j,j)|
    4. An updated Q'=Q X is computed, and this is now a random orthogonal matrix with a Haar measure.
    
    The implementation is a plain reimplementation/copy of the old m_randrot.f90 function.
*/



/**
   NB: This should rather use the implementation in m_mean_preserving_rotation.f90. 
*/

void enkf_analysis_set_randrot( matrix_type * Q ) {
  int ens_size       = matrix_get_rows( Q );
  double      * tau  = util_malloc( sizeof * tau  * ens_size , __func__);
  int         * sign = util_malloc( sizeof * sign * ens_size , __func__);

  for (int i = 0; i < ens_size; i++) 
    for (int j = 0; j < ens_size; j++) 
      matrix_iset(Q , i , j , enkf_util_rand_normal(0 , 1));

  matrix_dgeqrf( Q , tau );  /* QR factorization */
  for (int i=0; i  < ens_size; i++) {
    double Qii = matrix_iget( Q , i,i);
    sign[i] = Qii / abs(Qii);
  }

  matrix_dorgqr( Q , tau , ens_size );
  for (int i = 0; i < ens_size; i++) {
    if (sign[i] < 0)
      matrix_scale_column( Q , i , -1 );
  }

  free( sign );
  free( tau ); 
}


/**
   Generates the mean preserving random rotation for the EnKF SQRT algorithm
   using the algorithm from Sakov 2006-07.  I.e, generate rotation Up suceh that
   Up*Up^T=I and Up*1=1 (all rows have sum = 1)  see eq 17.
   From eq 18,    Up=B * Upb * B^T 
   B is a random orthonormal basis with the elements in the first column equals 1/sqrt(nrens)

   Upb = | 1  0 |
         | 0  U |

   where U is an arbitrary orthonormal matrix of dim nrens-1 x nrens-1  (eq. 19)
*/

matrix_type * enkf_analysis_alloc_mp_randrot(int ens_size ) {
  matrix_type * Up  = matrix_alloc( ens_size , ens_size );  /* The return value. */
  {
    matrix_type * B   = matrix_alloc( ens_size , ens_size );
    matrix_type * Upb = matrix_alloc( ens_size , ens_size );
    matrix_type * U   = matrix_alloc_shared(Upb , 1 , 1 , ens_size - 1, ens_size - 1);
    
    
    {
      int k,j;
      matrix_type * R   = matrix_alloc( ens_size , ens_size );
      matrix_random_init( B );   /* B is filled up with U(0,1) numbers. */
      matrix_set_const_column( B , 1.0 / sqrt( ens_size ) , 0 );

      /* modified_gram_schmidt is used to create the orthonormal basis in B.*/
      for (k=0; k < ens_size; k++) {
        double Rkk = sqrt( matrix_column_column_dot_product( B , k , B , k));   
        matrix_iset(R , k , k , Rkk);
        matrix_scale_column(B , k , 1.0/Rkk);
        for (j=k+1; j < ens_size; j++) {
          double Rkj = matrix_column_column_dot_product(B , k , B , j);
          matrix_iset(R , k , j , Rkj);
          {
            int i;
            for (i=0; i < ens_size; i++) {
              double Bij = matrix_iget(B , i , j);
              double Bik = matrix_iget(B , i , k);
              matrix_iset(B , i , j , Bij - Bik * Rkj);
            }
          }
        }
      }
      matrix_free( R );
    }
    
    enkf_analysis_set_randrot( U );
    matrix_iset( Upb , 0 , 0 , 1);
    
    
    {
      matrix_type * Q   = matrix_alloc( ens_size , ens_size );
      matrix_dgemm( Q  , B , Upb , false , false , 1, 0);   /* Q  = B * Ubp  */
      matrix_dgemm( Up , Q , B   , false , true  , 1, 0);   /* Up = Q * T(B) */
      matrix_free( Q );
    }
    
    matrix_free( B );
    matrix_free( Upb );
    matrix_free( U );
  }
  
  return Up;
}



/*****************************************************************/
/*****************************************************************/
/*                     High level functions                      */
/*****************************************************************/
/*****************************************************************/


void enkf_analysis_invertS(const analysis_config_type * config , const matrix_type * S , const matrix_type * R , matrix_type * W , double * eig) {
  double truncation                     = analysis_config_get_truncation( config );
  pseudo_inversion_type inversion_mode  = analysis_config_get_inversion_mode( config );  

  
  switch (inversion_mode) {
  case(SVD_SS_N1_R):
    lowrankCinv( S , R , W , eig , truncation );    
    break;
  default:
    util_abort("%s: inversion mode:%d not supported \n",__func__ , inversion_mode);
  }
}


/**
   Observe the following about the S matrix:

   1. On input the matrix is supposed to contain actual measurement values.
   2. On return the ensemble mean has been shifted away ...
   
*/


static void enkf_analysis_standard(matrix_type * X5 , const matrix_type * S , const matrix_type * D , const matrix_type * W , const double * eig) {
  const int nrobs   = matrix_get_rows( S );
  const int nrens   = matrix_get_columns( S );
  matrix_type * X3  = matrix_alloc(nrobs , nrens);
  
  genX3(X3 , W , D , eig );     
  matrix_dgemm( X5 , S , X3 , true , false , 1.0 , 0.0);  /* X5 = T(S) * X3 */
  {
    for (int i = 0; i < nrens; i++)
      matrix_iadd( X5 , i , i , 1.0);
  }
  
  matrix_free( X3 );
}



static void enkf_analysis_SQRT(matrix_type * X5 , const matrix_type * S , const matrix_type * randrot , const double * innov , const matrix_type * W , const double * eig) {
  const int nrobs   = matrix_get_rows( S );
  const int nrens   = matrix_get_columns( S );
  const int nrmin   = util_int_min( nrobs , nrens );
  
  matrix_type * X2    = matrix_alloc(nrmin , nrens);
  
  meanX5( S , W , eig , innov , X5 );
  genX2(X2 , S , W , eig);
  X5sqrt(X2 , X5 , randrot , nrobs);

  matrix_free( X2 );
}




/*****************************************************************/


void enkf_analysis_fprintf_obs_summary(const obs_data_type * obs_data , const meas_matrix_type * meas_matrix , int start_step, int end_step , const char * ministep_name , FILE * stream ) {
  const char * float_fmt = "%12.6f";
  int iobs;
  fprintf(stream , "======================================================================================================================\n");
  if (start_step == end_step)
    fprintf(stream , "Report step...: %04d \n",start_step);
  else
    fprintf(stream , "Report step...: %04d - %04d \n",start_step , end_step);
  
  fprintf(stream , "Ministep......: %s   \n",ministep_name);  
  fprintf(stream , "----------------------------------------------------------------------------------------------------------------------\n");
  if (obs_data_get_nrobs( obs_data ) > 0) {
    char * obs_fmt = util_alloc_sprintf("  %%-3d : %%-32s %s +/-  %s" , float_fmt , float_fmt);
    char * sim_fmt = util_alloc_sprintf("   %s +/- %s  \n"            , float_fmt , float_fmt);

    fprintf(stream , "                                                   Observed history               |             Simulated data        \n");  
    fprintf(stream , "----------------------------------------------------------------------------------------------------------------------\n");
    for (iobs = 0; iobs < obs_data_get_nrobs(obs_data); iobs++) {
      obs_data_node_type * node = obs_data_iget_node( obs_data , iobs);
      
      
      
      fprintf(stream , obs_fmt ,iobs + 1 , 
              obs_data_node_get_keyword(node),
              obs_data_node_get_value(node),
              obs_data_node_get_std(node));
      
      if (obs_data_node_active( node )) 
        fprintf(stream , "  Active   |");
      else
        fprintf(stream , "  Inactive |");
      {
        double mean,std;
        meas_matrix_iget_ens_mean_std( meas_matrix , iobs , &mean , &std);
        fprintf(stream , sim_fmt, mean , std);
      }
    }
    
    free( obs_fmt );
    free( sim_fmt );
  } else
    fprintf(stream , "No observations for this ministep / report_step. \n");
    
  fprintf(stream , "======================================================================================================================\n");
  fprintf(stream , "\n\n\n");
}




void enkf_analysis_deactivate_outliers(obs_data_type * obs_data , meas_matrix_type * meas_matrix , double std_cutoff , double alpha) {
  int nrobs = obs_data_get_nrobs( obs_data );
  int iobs;
  
  for (iobs = 0; iobs < nrobs; iobs++) {
    if (meas_matrix_iget_ens_std( meas_matrix , iobs) < std_cutoff) {
      /*
	De activated because the ensemble has to small variation for
	this particular measurement.
      */
      
      obs_data_deactivate_obs(obs_data , iobs , "No ensemble variation");
      meas_matrix_deactivate( meas_matrix , iobs );
    } else {
      double obs_value , obs_std;
      double ens_value , ens_std;
      double innov     ;
      
      obs_data_iget_value_std( obs_data , iobs , &obs_value , &obs_std);
      meas_matrix_iget_ens_mean_std( meas_matrix , iobs , &ens_value , &ens_std);
      innov = obs_value - ens_value;
      
      /* 
	 Deactivated because the distance between the observed data
	 and the ensemble prediction is to large. Keeping these
	 outliers will lead to numerical problems.
      */
      if (fabs( innov ) > alpha * (ens_std + obs_std)) {
	obs_data_deactivate_obs(obs_data , iobs , "No overlap");
	meas_matrix_deactivate(meas_matrix , iobs);
      }
    }
  }
}


/**
   Checking that the sum through one row in the X matrix is one.
*/

static void enkf_analysis_checkX(const matrix_type * X) {
  if ( !matrix_is_finite(X))
    util_abort("%s: The X matrix is not finite - internal meltdown. \n",__func__);
  {
    for (int icol = 0; icol < matrix_get_columns( X ); icol++) {
      double col_sum = matrix_get_column_sum(X , icol);
      if (fabs(col_sum - 1.0) > 0.0001) 
        util_abort("%s: something is seriously broken. col:%d  col_sum = %g != 1.0 - ABORTING\n",__func__ , icol , col_sum);
    }
  }
}





/**
   This function allocates a X matrix for the 

      A' = AX

   EnKF update. It takes as input a meas_matrix - where all the
   measurements have been collected, and a obs_data instance where the
   corresponding observations have been assembled. In addition it
   takes as input a random rotation matrix which will be used IFF the
   SQRT scheme is used, otherwise the random rotation matrix is not
   considered (and can be NULL).

   The function consists of three different parts:

    1. The function starts with several function call allocating the
       ordinary matrices S, D, R, E and innov.

    2. The S matrix is 'diagonalized', and singular vectors and
       singular values are stored in W and eig.

    3. The actual X matrix is calculated, based on either the standard
       enkf with perturbed measurement or the square root scheme.

*/
   
matrix_type * enkf_analysis_allocX( const analysis_config_type * config , meas_matrix_type * meas_matrix , obs_data_type * obs_data , const matrix_type * randrot) {
  int ens_size          = meas_matrix_get_ens_size( meas_matrix );
  matrix_type * X       = matrix_alloc( ens_size , ens_size );
  {
    int nrobs                = obs_data_get_active_size(obs_data);
    int nrmin                = util_int_min( ens_size , nrobs); 
    
    /*
      1: Allocating all matrices
    */
    matrix_type * S          = meas_matrix_allocS__( meas_matrix );
    matrix_type * R          = obs_data_allocR__(obs_data);
    matrix_type * W          = matrix_alloc(nrobs , nrmin);                      
    double      * eig        = util_malloc( sizeof * eig * nrmin , __func__);
    double      * innov      = obs_data_alloc_innov__(obs_data , meas_matrix);
    matrix_type * E          = NULL;
    matrix_type * D          = NULL;
    enkf_mode_type enkf_mode = analysis_config_get_enkf_mode( config );    
    
    
    if (enkf_mode == ENKF_STANDARD) {
      /* 
         We are using standard EnKF and need to perturbe the measurements,
         if we are using the SQRT scheme the E & D matrices are not used.
      */
      E = obs_data_allocE__(obs_data , ens_size);
      D = obs_data_allocD__(obs_data , E , S);
    } 
    
    obs_data_scale__(obs_data , S , E , D , R , innov );
    matrix_subtract_row_mean( S );  /* Subtracting the ensemble mean */
    
    /* 
       2: Diagonalize the S matrix; singular vectors are stored in W
          and singular values (after some massage) are stored in eig. 
    */
    enkf_analysis_invertS( config , S , R , W , eig );
    
    
    /* 
       3: actually calculating the X matrix. 
    */
    switch (enkf_mode) {
    case(ENKF_STANDARD):
      enkf_analysis_standard(X , S , D , W , eig);
      break;
    case(ENKF_SQRT):
      enkf_analysis_SQRT(X , S , randrot , innov , W , eig );
      break;
    default:
      util_abort("%s: INTERNAL ERROR \n",__func__);
    }
    
    matrix_free( W );
    matrix_free( R );
    matrix_free( S );
    free( innov );
    free( eig );
    
    if (enkf_mode == ENKF_STANDARD) {
      matrix_free( E );
      matrix_free( D );
    }
  }
  enkf_analysis_checkX(X);
  return X;
}
