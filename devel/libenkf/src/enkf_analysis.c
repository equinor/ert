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

  /* X1 = (I + Lambda1)^(-1) * W'*/
  for (i=0; i < nrmin; i++)
    for (j=0; j < nrobs; j++)
      matrix_iset(X1 , i , j , eig[i] * matrix_iget(W , j , i));
  
  matrix_matmul(X2 , X1 , D); /* X2 = X1 * D (Eq. 14.31) */
  matrix_matmul(X3 , W  , X2); /*  X3 = W * X2 = X1 * X2 (Eq. 14.31) */  
  
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



/* 
   Input is the S - matrix. U0 comes directly from dgesvd, sig0 has
   been seriously massaged. The right singular vectors of S are
   returned in V0T, this can be NULL.
   
   This method can be called in two slightly different ways.

     VOT == NULL: Truncation based on singular values (i.e. the
                  'traditional') way. In this case the value of the
                  truncation should be less than 1.0.

     VOT != NULL: This is used when Cross Validation will be
                  performed. In this case the truncation should be set
                  to 1.0

   The input parameters in this function are NOT orthogonal.
*/

static void svdS(const matrix_type * S , matrix_type * U0 , matrix_type * V0T , dgesvd_vector_enum jobVT , double * sig0, double truncation) {
  int num_singular_values = util_int_min( matrix_get_rows( S ) , matrix_get_columns( S ));
  {
    /* 
       The svd routine will destroy the contents of the input matrix,
       we therefor have to store a copy of S before calling it. The
       fortran implementation seems to handle this automagically??
    */
    matrix_type        * workS     = matrix_alloc_copy( S );
    matrix_dgesvd(DGESVD_MIN_RETURN , jobVT , workS , sig0 , U0 , V0T);  /* Have singular values in s0, , left hand singular vectors in U0 and right hand singular vector in V0T */
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

    /* Funny code ?? 
       Multiply B with S^(-1)from left and right
       BHat =  S^(-1) * B * S^(-1) 
    */
    for (j=0; j < matrix_get_columns( B ) ; j++)
      for (i=0; i < matrix_get_rows( B ); i++)
	matrix_imul(B , i , j , sig0[i]);

    for (j=0; j < matrix_get_columns( B ) ; j++)
      for (i=0; i < matrix_get_rows( B ); i++)
	matrix_imul(B , i , j , sig0[j]);
  }
  
  matrix_scale(B , nrens - 1.0);
}



int enkf_analysis_get_optimal_numb_comp(double * cvErr , int maxP) {

  int i, optP;
  double tmp, minErr;
  
  minErr = cvErr[0];
  optP = 1;

  for (i = 1; i < maxP; i++) {
    tmp = cvErr[i];
    if (tmp < minErr) {
      minErr = tmp;
      optP = i + 1;
    }
  }
    
  return optP;
}
  

/* function that estimates the Predictive Error Sum of Squares (PRESS)
   statistic based on k-fold Cross-Validation for a particular set of
   training and test indcies. Note that we do not need to recompute
   the eigenvalue decomposition X0 in equation 14.26 (Evensen, 2007)
   for each value of p.

   OUTPUT :

   cvErr      -  Vector containing the estimate PRESS for all valid combinations of p in the prediction

   INPUT :

   A          -  Matrix constaining the state vector ensemble
   VT         -  Matrix containing the transpose of the right singular vector of the S matrix 
   Z          -  Matrix containing the eigenvalues of the X0 matrix defined in Eq. 14.26 in Evensen (2007)
   eig        -  Vector containing the diagonal elements of the matrix L1i = inv(I + L1), where L1 
                  are the eigenvalues of the X0 matrix above
   indexTest  -  Vector containing integers specifying which ensemble members are
                  contained in the test ensemble
   indexTrain -  Vector containing integers specifying which ensemble members are 
                  contained in the training ensemble               
   nTest      -  Number of ensemble members in the test ensemble
   nTrain     -  Number of ensemble members in the training ensemble
*/


 
void enkf_analysis_get_cv_error(double * cvErr , const matrix_type * A , const matrix_type * VT , const matrix_type * Z , double * eig, int * indexTest, int * indexTrain , int nTest , int nTrain) { 
  /*  We need to predict ATest(p), for p = 1,...,nens -1, based on the estimated regression model:
     ATest(p) = A[:,indexTrain] * VT[1:p,indexTrain]'* Z[1:p,1:p] * eig[1:p,1:p] * Z[1:p,1:p]' * VT[1:p,testIndex]
  */

  /* Start by multiplying from the right: */
  int p,i,j,k;
  double tmp, tmp2;

  const int maxP = matrix_get_rows( VT );

  const int nx   = matrix_get_rows( A );
  

  printf("\n TestIndex = [");
  for (i = 0; i < nTest; i++) {
    printf(" %d ", indexTest[i]);
  }
  
  printf("]\n");

  printf("\n TrainIndex = [");
  for (i = 0; i < nTrain; i++) {
    printf(" %d ", indexTrain[i]);
  }
  
  printf("]\n");
  

  for (p = 0; p < maxP; p++) {
    
    printf("\n p = %d \n",p);
    matrix_type * W = matrix_alloc(p + 1 , nTest );
    matrix_type * W2 = matrix_alloc(p + 1 , nTest );
    matrix_type * W3 = matrix_alloc(nTrain, nTest );
    matrix_type * AHat = matrix_alloc(nx , nTest );
    
    /* Matrix multiplication: W = Z[1:p,1:p]' * VT[1:p,indexTest] */
    for (i = 0; i < p; i++) {
      for (j = 0; j < nTest; j++) {
        tmp = 0.0;
        for (k = 0; k < p; k++) {
          tmp += matrix_iget(Z , k , i) * matrix_iget(VT , k , indexTest[j]);
        }
        
        matrix_iset(W , i , j , tmp);
      }
    }

     
    /*Multiply W with the diagonal matrix eig[1:p,1:p] from the left */
    for (j=0; j < nTest; j++)
      for (i=0; i < p; i++)
	matrix_imul(W , i , j , eig[i]);
    
    for (i = 0; i < p; i++) {
      for (j = 0; j < nTest; j++) {
        tmp = 0.0;
        for (k = 0; k < p; k++) {
          tmp += matrix_iget(Z , i , k) * matrix_iget(W , k , j);
        }
        
        matrix_iset(W2 , i , j , tmp);
      }
    }

    matrix_free( W );

    
    /*Compute W3 = VT[TrainIndex,1:p]' * W*/
    for (i = 0; i < nTrain; i++) {
      for (j = 0; j < nTest; j++) {
        tmp = 0.0;
        for (k = 0; k < p; k++) {
          tmp += matrix_iget(VT , k , indexTrain[i] ) * matrix_iget(W2 , k , j);
        }
          
        matrix_iset(W3 , i , j , tmp);
      }
    }

    matrix_free( W2 );
    
    /*Compute AHat[:,indexTest](p) = A[:,indexTrain] * W */
    for (i = 0; i < nx; i++) {
      for (j = 0; j < nTest; j++) {
        tmp = 0.0;
        for (k = 0; k < nTrain; k++) {
          tmp += matrix_iget( A , i , indexTrain[k] ) * matrix_iget(W3 , k , j);
        }
          
        matrix_iset(AHat , i , j , tmp);
      }
    }

    matrix_free(W3);

    /*Compute Press Statistic: */
    tmp = 0.0;
    
    for (i = 0; i < nx; i++) {
      for (j = 0; j < nTest; j++) {
        tmp2 = matrix_iget(A , i , indexTest[j]) - matrix_iget(AHat , i , j);
        tmp += tmp2 * tmp2;
      }
    }
    
    cvErr[p] += tmp;
    
    matrix_free(AHat);
  } /*end for p */
}






static void lowrankCinv(const matrix_type * S , const matrix_type * R , matrix_type * W , double * eig , double truncation) {
  const int nrobs = matrix_get_rows( S );
  const int nrens = matrix_get_columns( S );
  const int nrmin = util_int_min( nrobs , nrens );

  matrix_type * B    = matrix_alloc( nrmin , nrmin );
  matrix_type * U0   = matrix_alloc( nrobs , nrmin );
  matrix_type * Z    = matrix_alloc( nrmin , nrmin );
  double * sig0      = util_malloc( nrmin * sizeof * sig0 , __func__);

  svdS(S , U0 , NULL /* V0T */ , DGESVD_NONE , sig0, truncation );
  lowrankCee( B , nrens , R , U0 , sig0);            /*B = Xo = (N-1) * Sigma0^(+) * U0'* Cee * U0 * Sigma0^(+')  (14.26)*/     
  matrix_dsyevx_all( DSYEVX_AUPPER , B , eig , Z);   /*(Eq. 14.27, Evensen, 2007)*/                       

  /*****************************************************************/
  {
    int i,j;
    /* Lambda1 = (I + Lambda)^(-1) */
    for (i=0; i < nrmin; i++) 
      eig[i] = 1.0 / (1 + eig[i]);
    
    for (j=0; j < nrmin; j++)
      for (i=0; i < nrmin; i++)
	matrix_imul(Z , i , j , sig0[i]); /* Z2 =  Sigma0^(+) * Z; */
  }
  /******************************************************************/

  matrix_matmul(W , U0 , Z); /* X1 = W = U0 * Z2 = U0 * Sigma0^(+') * Z    */
  util_safe_free( sig0 );
  matrix_free( U0 );
  matrix_free( B  );
  matrix_free( Z  );
}



static void lowrankCinv_pre_cv(const matrix_type * S , const matrix_type * R , matrix_type * V0T , matrix_type * Z, double * eig , matrix_type * U0) {
  const int nrobs = matrix_get_rows( S );
  const int nrens = matrix_get_columns( S );
  const int nrmin = util_int_min( nrobs , nrens );
  
  matrix_type * B    = matrix_alloc( nrmin , nrmin );
  double * sig0      = util_malloc( nrmin * sizeof * sig0 , __func__);

  /* No truncation initially */
  svdS(S , U0 , V0T , DGESVD_MIN_RETURN , sig0 , 1.0);
  lowrankCee( B , nrens , R , U0 , sig0);          /* B = Xo = (N-1) * Sigma0^(+) * U0'* Cee * U0 * Sigma0^(+')  (14.26)*/     
  matrix_dsyevx_all( DSYEVX_AUPPER , B , eig , Z); /* (Eq. 14.27, Evensen, 2007) */                       

  {
    int i,j;
    /* Lambda1 = (I + Lambda)^(-1) */
    for (i=0; i < nrmin; i++) 
      eig[i] = 1.0 / (1 + eig[i]);
    
    for (j=0; j < nrmin; j++)
      for (i=0; i < nrmin; i++)
	matrix_imul(Z , i , j , sig0[i]); /* Z2 =  Sigma0^(+) * Z; */
  }
}








/*Special function for doing cross-validation */ 
static void getW_pre_cv(matrix_type * W , matrix_type * V0T, matrix_type * Z , double * eig , matrix_type * U0 , int nfolds_CV, matrix_type * A) {

  const int nrobs = matrix_get_rows( U0 );
  const int nrens = matrix_get_columns( V0T );
  const int nrmin = util_int_min( nrobs , nrens );

  int i,j;
  
 
  /* Vector with random permutations of the itegers 1,...,nrens  */
  int * randperms     = util_malloc( sizeof * randperms * nrens, __func__);
  int * indexTest     = util_malloc( sizeof * indexTest * nrens, __func__);
  int * indexTrain    = util_malloc( sizeof * indexTrain * nrens, __func__);

  double * cvError    = util_malloc( sizeof * cvError * nrmin , __func__);
  
  /*Copy Z */
  matrix_type * workZ = matrix_alloc_copy( Z );

  int optP;
  
 
  /*printf("\n Size Z After eig decomp: (%d,%d)\n",matrix_get_rows(Z),matrix_get_columns(Z));
  printf("\n Size XO: (%d,%d)\n",matrix_get_rows(B),matrix_get_columns(B));
  printf("\n Size UO: (%d,%d)\n",matrix_get_rows(U0),matrix_get_columns(U0)); 
  printf("\n Size V0T: (%d,%d)\n",matrix_get_rows(V0T),matrix_get_columns(V0T));*/

 
  /* start cross-validation: */
  if ( nrens < nfolds_CV )
    util_abort("%s: number of ensemble members %d need to be larger than the number of cv-folds - aborting \n",__func__,nrens,nfolds_CV); 

  const int maxp = matrix_get_rows(V0T);
  
  /* draw random permutations of the integers 1,...,nrens */  
  enkf_util_randperm( randperms , nrens );
  
  /*need to init cvError to all zeros (?) */
  for (i = 0; i < nrmin; i++)
    cvError[i] = 0.0;
  
  
  int ntest, ntrain, k;
  for (i = 0; i < nfolds_CV; i++) {
    
    printf("\n computing cv error for test set %d\n",i + 1);
    
    ntest = 0;
    ntrain = 0;
    k = i;
    /*extract members for the training and test ensembles */
    for (j = 0; j < nrens; j++) {
      if (j == k) {
        indexTest[ntest] = randperms[j];
        k += nfolds_CV;
        ntest++;
      } else {
        indexTrain[ntrain] = randperms[j];
        ntrain++;
      }
          
    }
    
    
    enkf_analysis_get_cv_error( cvError , A , V0T , workZ , eig , indexTest , indexTrain, ntest, ntrain );
    
    
  }
  /* find optimal truncation value for the cv-scheme */
  optP = enkf_analysis_get_optimal_numb_comp( cvError , maxp);

  
  printf("\n optimal number of components found: %d \n",optP);
  FILE * compSel_log = util_fopen("compSel_log_local_cv" , "a");
  fprintf( compSel_log , " %d ",optP );
  fclose( compSel_log);

  

  /*free cvError vector and randperm */
  free( cvError );
  free( randperms );
  free( indexTest );
  free( indexTrain );

  /* need to update matrices so that we only use components 1,...,optP */
  /* remove non-zero entries of the z matrix (we do not want to recompute sigma0^(+') * z */
  /* this can surely be done much more efficiently, but for now we want to minimize the
     number of potential bugs in the code for now */ 
  for (i = optP; i < nrmin; i++) {
    for (j = 0; j < nrmin; j++) {
      matrix_iset(workZ , i , j, 0.0);
    }
  }
  
    
  /*fix the eig vector as well: */
  {
    int i;
    /* lambda1 = (i + lambda)^(-1) */
    for (i=optP; i < nrmin; i++) 
      eig[i] = 1.0;
  }

  matrix_matmul(W , U0 , workZ); /* x1 = w = u0 * z2 = u0 * sigma0^(+') * z    */
  

  matrix_free(workZ);
  /*end cross-validation */
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


/*Find optimal truncation factor based on k-fold CV */
//void enkf_analysis_invertS_cv(const analysis_config_type * config , const matrix_type * S , const matrix_type * R , matrix_type * W , double * eig, int nfolds_CV , matrix_type * A) {
//  pseudo_inversion_type inversion_mode  = analysis_config_get_inversion_mode( config );  
//  
//  switch (inversion_mode) {
//  case(SVD_SS_N1_R):
//    lowrankCinv_CV( S , R , W , eig , nfolds_CV , A);    
//    break;
//  default:
//    util_abort("%s: inversion mode:%d not supported \n",__func__ , inversion_mode);
//  }
//}



void enkf_analysis_invertS_pre_cv(const analysis_config_type * config , const matrix_type * S , const matrix_type * R , matrix_type * V0T , matrix_type * Z , double * eig , matrix_type * U0 ) {
  pseudo_inversion_type inversion_mode  = analysis_config_get_inversion_mode( config );  

  
  switch (inversion_mode) {
  case(SVD_SS_N1_R):
    lowrankCinv_pre_cv( S , R , V0T , Z , eig , U0);    
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
  
  genX3(X3 , W , D , eig ); /*  X2 = diag(eig) * W' * D (Eq. 14.31, Evensen (2007)) */
                            /*  X3 = W * X2 = X1 * X2 (Eq. 14.31, Evensen (2007)) */  

  matrix_dgemm( X5 , S , X3 , true , false , 1.0 , 0.0);  /* X5 = T(S) * X3 */
  {
    for (int i = 0; i < nrens; i++)
      matrix_iadd( X5 , i , i , 1.0); /*X5 = I + X5 */
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
  const char * float_fmt = "%15.3f";
  int iobs;
  fprintf(stream , "===============================================================================================================================\n");
  if (start_step == end_step)
    fprintf(stream , "Report step...: %04d \n",start_step);
  else
    fprintf(stream , "Report step...: %04d - %04d \n",start_step , end_step);
  
  fprintf(stream , "Ministep......: %s   \n",ministep_name);  
  fprintf(stream , "-------------------------------------------------------------------------------------------------------------------------------\n");
  if (obs_data_get_nrobs( obs_data ) > 0) {
    char * obs_fmt = util_alloc_sprintf("  %%-3d : %%-32s %s +/-  %s" , float_fmt , float_fmt);
    char * sim_fmt = util_alloc_sprintf("   %s +/- %s  \n"            , float_fmt , float_fmt);

    fprintf(stream , "                                                         Observed history               |             Simulated data        \n");  
    fprintf(stream , "-------------------------------------------------------------------------------------------------------------------------------\n");
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
    
  fprintf(stream , "===============================================================================================================================\n");
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



/*
  This function will allocate and initialize the matrices S,R,D and E
  and also the innovation. The matrices will be scaled with the
  observation error and the mean will be subtracted from the S
  matrix. 
*/

static void enkf_analysis_alloc_matrices( const meas_matrix_type * meas_matrix , const obs_data_type * obs_data , enkf_mode_type enkf_mode , 
                                          matrix_type ** S , 
                                          matrix_type ** R , 
                                          double      ** innov,
                                          matrix_type ** E ,
                                          matrix_type ** D ) {
  int ens_size = meas_matrix_get_ens_size( meas_matrix );     
  *S           = meas_matrix_allocS( meas_matrix );
  *R           = obs_data_allocR(obs_data);
  *innov       = obs_data_alloc_innov(obs_data , meas_matrix);
  
  if (enkf_mode == ENKF_STANDARD) {
    /* 
       We are using standard EnKF and need to perturbe the measurements,
       if we are using the SQRT scheme the E & D matrices are not used.
    */
    *E = obs_data_allocE(obs_data , ens_size);
    *D = obs_data_allocD(obs_data , *E , *S);
  } else {
    *E          = NULL;
    *D          = NULL;
  }
  
  obs_data_scale(obs_data , *S , *E , *D , *R , *innov );
  matrix_subtract_row_mean( *S );  /* Subtracting the ensemble mean */
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
    matrix_type * S , *R , *E , *D;
    double      * innov;
    int nrobs                = obs_data_get_active_size(obs_data);
    int nrmin                = util_int_min( ens_size , nrobs); 

    matrix_type * W          = matrix_alloc(nrobs , nrmin);                      
    double      * eig        = util_malloc( sizeof * eig * nrmin , __func__);    
    enkf_mode_type enkf_mode = analysis_config_get_enkf_mode( config );    
    enkf_analysis_alloc_matrices( meas_matrix , obs_data , enkf_mode , &S , &R , &innov , &E , &D );
    
        
    /* 
       2: Diagonalize the S matrix; singular vectors are stored in W
          and singular values (after some massage) are stored in eig. 
          W = X1, eig = inv(I+Lambda1),(Eq.14.30, and 14.29, Evensen, 2007, respectively)
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



matrix_type * enkf_analysis_allocX_pre_cv( const analysis_config_type * config , meas_matrix_type * meas_matrix , obs_data_type * obs_data , const matrix_type * randrot , matrix_type * A , matrix_type * V0T , matrix_type * Z , double * eig , matrix_type * U0) {
  int ens_size          = meas_matrix_get_ens_size( meas_matrix );
  matrix_type * X       = matrix_alloc( ens_size , ens_size );
  {
    int nrobs                = obs_data_get_active_size(obs_data);
    int nrmin                = util_int_min( ens_size , nrobs); 
    int nfolds_CV            = analysis_config_get_nfolds_CV( config );
    
    /*
      1: Allocating all matrices
    */
    /*Need a copy of A, because we need it later */
    matrix_type * workA      = matrix_alloc_copy( A );    /* <- This is a massive memory requirement. */
    matrix_type * S , *R , *E , *D;
    double      * innov;
    
    matrix_type * W          = matrix_alloc(nrobs , nrmin);                      
    enkf_mode_type enkf_mode = analysis_config_get_enkf_mode( config );    
    
    double * workeig    = util_malloc( sizeof * workeig * nrmin , __func__);

    enkf_analysis_alloc_matrices( meas_matrix , obs_data , enkf_mode , &S , &R , &innov , &E , &D );

    /*copy entries in eig:*/
    {
      int i;
      for (i = 0 ; i < nrmin ; i++) 
        workeig[i] = eig[i];
    }
    
    /* Subtracting the ensemble mean of the state vector ensemble */
    matrix_subtract_row_mean( workA );
    
    /* 
       2: Diagonalize the S matrix; singular vectors are stored in W
          and singular values (after some massage) are stored in eig. 
    W = X1, eig = inv(I+Lambda1),(Eq.14.30, and 14.29, Evensen, 2007, respectively)
    */ 

    getW_pre_cv(W , V0T , Z , workeig ,  U0 , nfolds_CV , workA);
    
    /* 
       3: actually calculating the X matrix. 
    */
    switch (enkf_mode) {
    case(ENKF_STANDARD):
      enkf_analysis_standard(X , S , D , W , workeig);
      break;
    case(ENKF_SQRT):
      enkf_analysis_SQRT(X , S , randrot , innov , W , workeig );
      break;
    default:
      util_abort("%s: INTERNAL ERROR \n",__func__);
    }
    
    matrix_free( W );
    matrix_free( R );
    matrix_free( S );
    matrix_free( workA );
    free( innov );
    free( workeig );
    
    if (enkf_mode == ENKF_STANDARD) {
      matrix_free( E );
      matrix_free( D );
    }
  }
  enkf_analysis_checkX(X);
  return X;
}



/** 
    This function initializes the S matrix and performs svd(S). The
    left and right singular vectors and singular values are returned
    in U0, V0T and eig respectively.
*/

void enkf_analysis_local_pre_cv( const analysis_config_type * config , meas_matrix_type * meas_matrix , obs_data_type * obs_data ,  matrix_type * V0T , matrix_type * Z , double * eig , matrix_type * U0) {
  {
    matrix_type * S , *R , *E , *D;
    double      * innov;

    enkf_mode_type enkf_mode = analysis_config_get_enkf_mode( config );    
    enkf_analysis_alloc_matrices( meas_matrix , obs_data , enkf_mode , &S , &R , &innov , &E , &D );
    
    /* 
       2: Diagonalize the S matrix; singular vectors etc. needed later in the local CV:
       (V0T = transposed right singular vectors of S, Z = scaled principal components, 
       eig = scaled, inverted singular vectors, U0 = left singular vectors of S
       , eig = inv(I+Lambda1),(Eq.14.30, and 14.29, Evensen, 2007, respectively)
    */ 
    enkf_analysis_invertS_pre_cv( config , S , R , V0T , Z , eig , U0);
    
    
    matrix_free( R );
    matrix_free( S );
    free( innov );
        
    if (enkf_mode == ENKF_STANDARD) {
      matrix_free( E );
      matrix_free( D );

    }
  }
  
}


