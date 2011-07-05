/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'enkf_analysis.c' is part of ERT - Ensemble based Reservoir Tool. 
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
*/

#include <util.h>
#include <math.h>
#include <matrix.h>
#include <matrix_lapack.h>
#include <matrix_blas.h>
#include <meas_data.h>
#include <obs_data.h>
#include <analysis_config.h>
#include <enkf_util.h>
#include <enkf_analysis.h>
#include <timer.h>
#include <rng.h>

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
  
  matrix_matmul(X2 , X1 , D); /*   X2 = X1 * D           (Eq. 14.31) */
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
    double total_sigma2    = 0.0;
    double running_sigma2  = 0.0;
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

    printf("Subspace dimension selected based on a truncation factor of %0.2f : %d\n",truncation,num_significant);

    /* Explicitly setting the insignificant singular values to zero. */
    for (i=num_significant; i < num_singular_values; i++)
      sig0[i] = 0;                                     
    
    /* Inverting the significant singular values */
    for (i = 0; i < num_significant; i++)
      sig0[i] = 1.0 / sig0[i];
  }
}


static void svdS_force(const matrix_type * S , matrix_type * U0 , matrix_type * V0T , dgesvd_vector_enum jobVT , double * sig0, int ncomp) {
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
    int    num_significant = ncomp;
    int i;
    for (i=0; i < num_singular_values; i++)
      total_sigma2 += sig0[i] * sig0[i];
    
    for (i=0; i < ncomp; i++) {
      running_sigma2 += sig0[i] * sig0[i];
    }
    
    printf("Selected %d components. Variance explained = %f \n",ncomp,running_sigma2 / total_sigma2);


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



int enkf_analysis_get_optimal_numb_comp(const matrix_type * cvErr , const int maxP , const int nFolds , const bool pen_press) {

  int i, optP;
  double tmp, minErr;
  
  double tmp2 = (1.0 / (double)nFolds); 

  double * cvMean = util_malloc( sizeof * cvMean * maxP, __func__);
  
  for (int p = 0; p < maxP; p++ ){
    tmp = 0.0;
    for (int folds = 0; folds < nFolds; folds++ ){
      tmp += matrix_iget( cvErr , p, folds );
    }
    cvMean[p] = tmp * tmp2;
  }

  
  tmp2 = 1.0 / ((double)(nFolds - 1));
  double * cvStd = util_malloc( sizeof * cvStd * maxP, __func__);
  for ( int p = 0; p < maxP; p++){
    tmp = 0.0;
    for ( int folds = 0; folds < nFolds; folds++){
      tmp += pow( matrix_iget( cvErr , p , folds ) - cvMean[p] , 2);
    }
    cvStd[p] = sqrt( tmp * tmp2 );
  }

  minErr = cvMean[0];
  optP = 1;
  

  printf("PRESS = \n");
  for (i = 0; i < maxP; i++) {
    printf(" %0.2f \n",cvMean[i]);
  }
  


  for (i = 1; i < maxP; i++) {
    tmp = cvMean[i];
    if (tmp < minErr && tmp > 0.0) {
      minErr = tmp;
      optP = i+1;
    }
  }

  printf("Global optimum= %d\n",optP);
  

  if (pen_press) {
    printf("Selecting optimal number of components using Penalised PRESS statistic: \n");
    for ( i = 0; i < optP; i++){
      if( cvMean[i] - cvStd[i] <= minErr ){
	optP = i+1;
	break;
      }
    }
  }
  

  free( cvStd );
  free( cvMean );
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


 
static void enkf_analysis_get_cv_error(matrix_type * cvErr , const matrix_type * A , const matrix_type * VT , const matrix_type * Z , const double * eig, const int * indexTest, const int * indexTrain , const int nTest , const int nTrain , const int foldIndex) { 
  /*  We need to predict ATest(p), for p = 1,...,nens -1, based on the estimated regression model:
     ATest(p) = A[:,indexTrain] * VT[1:p,indexTrain]'* Z[1:p,1:p] * eig[1:p,1:p] * Z[1:p,1:p]' * VT[1:p,testIndex]
  */
  
  /* Start by multiplying from the right: */
  int p,i,j,k;
  double tmp, tmp2;

  int maxP = matrix_get_rows( VT );

  const int nx   = matrix_get_rows( A );

  /* We only want to search the non-zero eigenvalues */
  for (i = 0; i < maxP; i++) {
    if (eig[i] == 1.0) {
      maxP = i;
      break;
    }
  }

  matrix_type * AHat = matrix_alloc(nx , nTest );
  matrix_type * W3 = matrix_alloc(nTrain, nTest );
    
  /*We want to use the blas function to speed things up: */
  matrix_type * ATrain = matrix_alloc( nx , nTrain );
  /* Copy elements*/
  for (i = 0; i < nx; i++) {
    for (j = 0; j < nTrain; j++) {
      matrix_iset(ATrain , i , j , matrix_iget( A , i , indexTrain[j]));
    }
  }
  


  for (p = 0; p < maxP; p++) {
    
      
    
    /*    printf("p = %d \n",p);*/
    matrix_type * W = matrix_alloc(p + 1 , nTest );
    matrix_type * W2 = matrix_alloc(p + 1 , nTest );
    
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
    

    matrix_matmul(AHat , ATrain , W3 );



    /*Compute Press Statistic: */
    tmp = 0.0;
    
    for (i = 0; i < nx; i++) {
      for (j = 0; j < nTest; j++) {
        tmp2 = matrix_iget(A , i , indexTest[j]) - matrix_iget(AHat , i , j);
        tmp += tmp2 * tmp2;
      }
    }
    
    matrix_iset( cvErr , p , foldIndex , tmp );
    
  } /*end for p */
  
  matrix_free( AHat );
  matrix_free( ATrain );
  matrix_free( W3 );

}


/*Function that computes the PRESS for different subspace dimensions using
  m-fold CV 
  INPUT :
  A   : State-Vector ensemble matrix
  Z   : Ensemble matrix of principal components
  Rp  : Reduced order Observation error matrix
  indexTrain: index of training ensemble
  indexTest: index of test ensemble
  nTest : number of members in the training ensemble
  nTrain . number of members in the test ensemble
  foldIndex: integer specifying which "cv-fold" we are considering
  
  OUTPUT:
  cvErr : UPDATED MATRIX OF PRESS VALUES
  
*/
static void enkf_analysis_get_cv_error_prin_comp(matrix_type * cvErr , const matrix_type * A , const matrix_type * Z , const matrix_type * Rp , const int * indexTest, const int * indexTrain , const int nTest , const int nTrain , const int foldIndex, const int maxP) { 
  /*  We need to predict ATest(p), for p = 1,...,nens -1, based on the estimated regression model:
     AHatTest(p) = A[:,indexTrain] * Z[1:p,indexTrain]'* inv( Z[1:p,indexTrain] * Z[1:p,indexTrain]' + (nens-1) * Rp[1:p,1:p] ) * Z[1:p,indexTest];
  */
  
  /* Start by multiplying from the right: */
  int p,i,j,k, inv_ok, tmp3;
  double tmp, tmp2;



  const int nx   = matrix_get_rows( A );


  matrix_type * AHat = matrix_alloc(nx , nTest );
    
  /*We want to use the blas function to speed things up: */
  matrix_type * ATrain = matrix_alloc( nx , nTrain );
  /* Copy elements*/
  for (i = 0; i < nx; i++) {
    for (j = 0; j < nTrain; j++) {
      matrix_iset(ATrain , i , j , matrix_iget( A , i , indexTrain[j]));
    }
  }
  
  tmp3 = nTrain - 1;
  int pOrg;

  for (p = 0; p < maxP; p++) {

     pOrg = p + 1;


    /*For now we do this the hard way through a full inversion of the reduced data covariance matrix: */
    /* Alloc ZTrain(1:p): */
    matrix_type *ZpTrain = matrix_alloc( pOrg, nTrain );
    for (i = 0; i < pOrg ; i++) {
      for (j = 0; j < nTrain; j++) {
	matrix_iset(ZpTrain , i , j , matrix_iget(Z , i ,indexTrain[j]));
      }
    }

    matrix_type *SigDp = matrix_alloc( pOrg ,pOrg);
    /*Compute SigDp = ZpTrain * ZpTrain' */
    matrix_dgemm( SigDp , ZpTrain , ZpTrain, false , true , 1.0, 0.0);
    
    /*Add (ntrain-1) * Rp*/

    for(i = 0; i < pOrg; i++) {
      for( j = 0; j < pOrg; j++) {
	tmp2 = matrix_iget(SigDp , i , j) + tmp3 * matrix_iget(Rp, i, j);
	matrix_iset( SigDp , i , j , tmp2 );
      }
    }
    
    /* Invert the covariance matrix for the principal components  */
    inv_ok = matrix_inv( SigDp );

    
    /*Check if the inversion went ok */
    if ( inv_ok != 0 ) {
      util_abort("%s: inversion of covariance matrix for the principal components failed for subspace dimension p = %d\n - aborting \n",__func__,pOrg); 
    }

    
    /*Compute inv(SigDp) * ZTest: */
    matrix_type * W = matrix_alloc(pOrg , nTest );
    for (i = 0; i < pOrg; i++) {
      for (j = 0; j < nTest; j++) {
	tmp = 0.0;
        for (k = 0; k < pOrg; k++) {
	  tmp += matrix_iget(SigDp , i , k) * matrix_iget(Z , k , indexTest[j]);
        }
	matrix_iset(W , i , j , tmp);
      }
    }



    matrix_type * W2 = matrix_alloc(nTrain , nTest );
    /*Compute W2 = ZpTrain' * W */
    matrix_dgemm( W2 , ZpTrain , W , true , false , 1.0 , 0.0);
    
    matrix_free( ZpTrain );
    matrix_free( SigDp );
    matrix_free( W );

    /*Estimate the state-vector */
    matrix_matmul(AHat , ATrain , W2 );
    matrix_free( W2 );

    /*Compute Press Statistic: */
    tmp = 0.0;
    
    for (i = 0; i < nx; i++) {
      for (j = 0; j < nTest; j++) {
        tmp2 = matrix_iget(A , i , indexTest[j]) - matrix_iget(AHat , i , j);
        tmp += tmp2 * tmp2;
      }
    }
    
    matrix_iset( cvErr , p , foldIndex , tmp );
    
  } /*end for p */
  
  matrix_free( AHat );
  matrix_free( ATrain );
}






static void lowrankCinv(const matrix_type * S , 
                        const matrix_type * R , 
                        matrix_type * W       , /* Corresponding to X1 from Eq. 14.29 */
                        double * eig          , /* Corresponding to 1 / (1 + Lambda_1) (14.29) */
                        double truncation) {

  const int nrobs = matrix_get_rows( S );
  const int nrens = matrix_get_columns( S );
  const int nrmin = util_int_min( nrobs , nrens );

  matrix_type * B    = matrix_alloc( nrmin , nrmin );
  matrix_type * U0   = matrix_alloc( nrobs , nrmin );
  matrix_type * Z    = matrix_alloc( nrmin , nrmin );
  double * sig0      = util_malloc( nrmin * sizeof * sig0 , __func__);

  svdS(S , U0 , NULL /* V0T */ , DGESVD_NONE , sig0, truncation );
  lowrankCee( B , nrens , R , U0 , sig0);            /* B = Xo = (N-1) * Sigma0^(+) * U0'* Cee * U0 * Sigma0^(+')  (14.26)*/     
  matrix_dsyevx_all( DSYEVX_AUPPER , B , eig , Z);   /*(Eq. 14.27, Evensen, 2007)*/                       

  /*****************************************************************/
  {
    int i,j;

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

/* HERE WE FORCE THE NUMBER OF PRINCIPAL COMPONENTS USED IN THE REGRESSION */
static void lowrankCinv_force(const matrix_type * S , 
                        const matrix_type * R , 
                        matrix_type * W       , /* Corresponding to X1 from Eq. 14.29 */
                        double * eig          , /* Corresponding to 1 / (1 + Lambda_1) (14.29) */
                        int ncomp) {

  const int nrobs = matrix_get_rows( S );
  const int nrens = matrix_get_columns( S );
  const int nrmin = util_int_min( nrobs , nrens );

  matrix_type * B    = matrix_alloc( nrmin , nrmin );
  matrix_type * U0   = matrix_alloc( nrobs , nrmin );
  matrix_type * Z    = matrix_alloc( nrmin , nrmin );
  double * sig0      = util_malloc( nrmin * sizeof * sig0 , __func__);

  svdS_force(S , U0 , NULL /* V0T */ , DGESVD_NONE , sig0, ncomp );
  lowrankCee( B , nrens , R , U0 , sig0);            /* B = Xo = (N-1) * Sigma0^(+) * U0'* Cee * U0 * Sigma0^(+')  (14.26)*/     


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



static void lowrankCinv_pre_cv(const matrix_type * S , const matrix_type * R , matrix_type * V0T , matrix_type * Z, double * eig , matrix_type * U0, double truncation) {
  const int nrobs = matrix_get_rows( S );
  const int nrens = matrix_get_columns( S );
  const int nrmin = util_int_min( nrobs , nrens );
  
  matrix_type * B    = matrix_alloc( nrmin , nrmin );
  double * sig0      = util_malloc( nrmin * sizeof * sig0 , __func__);

  svdS(S , U0 , V0T , DGESVD_MIN_RETURN , sig0 , truncation);
  lowrankCee( B , nrens , R , U0 , sig0);          /* B = Xo = (N-1) * Sigma0^(+) * U0'* Cee * U0 * Sigma0^(+')  (14.26)*/     
  
  /*USE SVD INSTEAD*/
  matrix_dgesvd(DGESVD_MIN_RETURN , DGESVD_NONE, B , eig, Z , NULL);

  matrix_free( B );

  
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
static void getW_pre_cv(matrix_type * W , const matrix_type * V0T, const matrix_type * Z , double * eig , const matrix_type * U0 , int nfolds_CV, 
			const matrix_type * A, int unique_bootstrap_components , rng_type * rng, bool pen_press) {

  const int nrobs = matrix_get_rows( U0 );
  const int nrens = matrix_get_columns( V0T );
  const int nrmin = util_int_min( nrobs , nrens );

  int i,j;
  
 
  /* Vector with random permutations of the itegers 1,...,nrens  */
  int * randperms     = util_malloc( sizeof * randperms * nrens, __func__);
  int * indexTest     = util_malloc( sizeof * indexTest * nrens, __func__);
  int * indexTrain    = util_malloc( sizeof * indexTrain * nrens, __func__);

  if(nrens != unique_bootstrap_components)
    nfolds_CV = util_int_min( nfolds_CV , unique_bootstrap_components-1);


  matrix_type * cvError = matrix_alloc( nrmin,nfolds_CV );
  
  /*Copy Z */
  matrix_type * workZ = matrix_alloc_copy( Z );

  int optP;
  
 
  /* start cross-validation: */

  const int maxp = matrix_get_rows(V0T);
  
  /* draw random permutations of the integers 1,...,nrens */  
  enkf_util_randperm( randperms , nrens , rng);
  
  /*need to init cvError to all zeros (?) */
  for (i = 0; i < nrmin; i++){
    for( j = 0; j> nfolds_CV; j++){
      matrix_iset( cvError , i , j , 0.0 );
    }
  }
  
  int ntest, ntrain, k;
  printf("\nStarting cross-validation\n");
  for (i = 0; i < nfolds_CV; i++) {
    printf(".");

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
    enkf_analysis_get_cv_error( cvError , A , V0T , workZ , eig , indexTest , indexTrain, ntest, ntrain , i );
  }
  printf("\n");
  /* find optimal truncation value for the cv-scheme */
  optP = enkf_analysis_get_optimal_numb_comp( cvError , maxp, nfolds_CV, pen_press);

  printf("Optimal number of components found: %d \n",optP);
  printf("\n");
  FILE * compSel_log = util_fopen("compSel_log_local_cv" , "a");
  fprintf( compSel_log , " %d ",optP);
  fclose( compSel_log);


  /*free cvError vector and randperm */
  matrix_free( cvError );
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
  



  /*end cross-validation */
}


/* Function that performs cross-validation to find the optimal subspace dimension,  */


int get_optimal_principal_components(const matrix_type * Z , const matrix_type * Rp , int nfolds_CV, const matrix_type * A, rng_type * rng, const int maxP, bool pen_press) {

  const int nrens = matrix_get_columns( Z );
  const int nrmin = matrix_get_rows( Z );

  int i,j;
  
 
  /* Vector with random permutations of the itegers 1,...,nrens  */
  int * randperms     = util_malloc( sizeof * randperms * nrens, __func__);
  int * indexTest     = util_malloc( sizeof * indexTest * nrens, __func__);
  int * indexTrain    = util_malloc( sizeof * indexTrain * nrens, __func__);


  if ( nrens < nfolds_CV )
    util_abort("%s: number of ensemble members %d need to be larger than the number of cv-folds - aborting \n",__func__,nrens,nfolds_CV); 
  
  
  
  int optP;


  printf("\nOnly searching for the optimal subspace dimension among the first %d principal components\n",maxP);
  
  matrix_type * cvError = matrix_alloc( maxP ,nfolds_CV );

 
  /* start cross-validation: */
  if ( nrens < nfolds_CV )
    util_abort("%s: number of ensemble members %d need to be larger than the number of cv-folds - aborting \n",__func__,nrens,nfolds_CV); 

  
  /* draw random permutations of the integers 1,...,nrens */  
  enkf_util_randperm( randperms , nrens , rng);
  
  /*need to init cvError to all zeros (?) */
  for (i = 0; i < nrmin; i++){
    for( j = 0; j> nfolds_CV; j++){
      matrix_iset( cvError , i , j , 0.0 );
    }
  }
  
  int ntest, ntrain, k;
  printf("Starting cross-validation\n");
  for (i = 0; i < nfolds_CV; i++) {
    printf(".");

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
    
    /*Perform CV for each subspace dimension p */
    enkf_analysis_get_cv_error_prin_comp( cvError , A , Z , Rp , indexTest , indexTrain, ntest, ntrain , i , maxP);
  }
  printf("\n");
  /* find optimal truncation value for the cv-scheme */
  optP = enkf_analysis_get_optimal_numb_comp( cvError , maxP, nfolds_CV , pen_press);

  printf("Optimal number of components found: %d \n",optP);
  FILE * compSel_log = util_fopen("compSel_log_local_cv" , "a");
  fprintf( compSel_log , " %d ",optP);
  fclose( compSel_log);


  /*free cvError vector and randperm */
  matrix_free( cvError );
  free( randperms );
  free( indexTest );
  free( indexTrain );


  return optP;
}


/*NB! HERE WE COUNT optP from 0,1,2,... */
static void getW_prin_comp(matrix_type *W , const matrix_type * Z , 
			   const matrix_type * Rp , const int optP) { 

  int i, j;
  double tmp2;
  int nrens = matrix_get_columns( Z );
  
  /* Finally, compute W = Z(1:p,:)' * inv(Z(1:p,:) * Z(1:p,:)' + (n -1) * Rp) */
  matrix_type *Zp = matrix_alloc( optP, nrens );
  for (i = 0; i < optP ; i++) {
    for (j = 0; j < nrens; j++) {
      matrix_iset(Zp , i , j , matrix_iget(Z , i ,j));
    }
  }

  matrix_type *SigZp = matrix_alloc( optP ,optP);
  /*Compute SigZp = Zp * Zp' */
  matrix_dgemm( SigZp , Zp , Zp, false , true , 1.0, 0.0);
  
  /*Add (ntrain-1) * Rp*/
  
  int tmp3 = nrens - 1;


  for(i = 0; i < optP; i++) {
    for( j = 0; j < optP; j++) {
      tmp2 = matrix_iget(SigZp , i , j) + tmp3 * matrix_iget(Rp, i, j);
      matrix_iset( SigZp , i , j , tmp2 );
    }
  }
  
  /* Invert the covariance matrix for the principal components  */
  int inv_ok = matrix_inv( SigZp );
  
  /*Check if the inversion went ok */
  if ( inv_ok != 0 ) {
    util_abort("%s: inversion of covariance matrix for the principal components failed for subspace dimension p = %d\n - aborting \n",__func__,optP); 
  }
  


  
  /*Compute W = Zp' * inv(SigZp) */
  matrix_dgemm( W , Zp , SigZp , true , false , 1.0 , 0.0);

  matrix_free( Zp );
  matrix_free( SigZp );
  

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

void enkf_analysis_set_randrot( matrix_type * Q  , rng_type * rng) {
  int ens_size       = matrix_get_rows( Q );
  double      * tau  = util_malloc( sizeof * tau  * ens_size , __func__);
  int         * sign = util_malloc( sizeof * sign * ens_size , __func__);

  for (int i = 0; i < ens_size; i++) 
    for (int j = 0; j < ens_size; j++) 
      matrix_iset(Q , i , j , enkf_util_rand_normal(0 , 1 , rng));

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

matrix_type * enkf_analysis_alloc_mp_randrot(int ens_size , rng_type * rng) {
  matrix_type * Up  = matrix_alloc( ens_size , ens_size );  /* The return value. */
  {
    matrix_type * B   = matrix_alloc( ens_size , ens_size );
    matrix_type * Upb = matrix_alloc( ens_size , ens_size );
    matrix_type * U   = matrix_alloc_shared(Upb , 1 , 1 , ens_size - 1, ens_size - 1);
    
    
    {
      int k,j;
      matrix_type * R   = matrix_alloc( ens_size , ens_size );
      matrix_random_init( B , rng);   /* B is filled up with U(0,1) numbers. */
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
    
    enkf_analysis_set_randrot( U , rng );
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


void enkf_analysis_invertS_force(const analysis_config_type * config , const matrix_type * S , const matrix_type * R , matrix_type * W , double * eig, int ncomp) {
  pseudo_inversion_type inversion_mode  = analysis_config_get_inversion_mode( config );  
  
  switch (inversion_mode) {
  case(SVD_SS_N1_R):
    lowrankCinv_force( S , R , W , eig , ncomp );    
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
  double truncation                     = analysis_config_get_truncation( config );  
  
  switch (inversion_mode) {
  case(SVD_SS_N1_R):
    lowrankCinv_pre_cv( S , R , V0T , Z , eig , U0 , truncation);    
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


static void enkf_analysis_standard(matrix_type * X5 , const matrix_type * S , const matrix_type * D , const matrix_type * W , const double * eig, bool bootstrap) {
  const int nrobs   = matrix_get_rows( S );
  const int nrens   = matrix_get_columns( S );
  matrix_type * X3  = matrix_alloc(nrobs , nrens);
  
  genX3(X3 , W , D , eig ); /*  X2 = diag(eig) * W' * D (Eq. 14.31, Evensen (2007)) */
                            /*  X3 = W * X2 = X1 * X2 (Eq. 14.31, Evensen (2007)) */  

  matrix_dgemm( X5 , S , X3 , true , false , 1.0 , 0.0);  /* X5 = S' * X3 */
  if (!bootstrap) {
    for (int i = 0; i < nrens; i++)
      matrix_iadd( X5 , i , i , 1.0); /*X5 = I + X5 */
  }
  
  matrix_free( X3 );
}


static void enkf_analysis_SQRT(matrix_type * X5 , const matrix_type * S , const matrix_type * randrot , const double * innov , const matrix_type * W , const double * eig , bool bootstrap) {
  const int nrobs   = matrix_get_rows( S );
  const int nrens   = matrix_get_columns( S );
  const int nrmin   = util_int_min( nrobs , nrens );
  
  matrix_type * X2    = matrix_alloc(nrmin , nrens);
  
  if (bootstrap)
    util_exit("%s: Sorry bootstrap support not fully implemented for SQRT scheme\n",__func__);

  meanX5( S , W , eig , innov , X5 );
  genX2(X2 , S , W , eig);
  X5sqrt(X2 , X5 , randrot , nrobs);

  matrix_free( X2 );
}




/*****************************************************************/

void enkf_analysis_fprintf_obs_summary(const obs_data_type * obs_data , const meas_data_type * meas_data , const int_vector_type * step_list , const char * ministep_name , FILE * stream ) {
  const char * float_fmt = "%15.3f";
  fprintf(stream , "===============================================================================================================================\n");
  fprintf(stream , "Report step...: %04d",int_vector_iget( step_list , 0));
  if (int_vector_size( step_list ) == 1)
    fprintf(stream , "\n");
  else
    fprintf(stream , " - %04d \n",int_vector_get_last( step_list ));
  
  
  fprintf(stream , "Ministep......: %s   \n",ministep_name);  
  fprintf(stream , "-------------------------------------------------------------------------------------------------------------------------------\n");
  {
    char * obs_fmt = util_alloc_sprintf("  %%-3d : %%-32s %s +/-  %s" , float_fmt , float_fmt);
    char * sim_fmt = util_alloc_sprintf("   %s +/- %s  \n"            , float_fmt , float_fmt);

    fprintf(stream , "                                                         Observed history               |             Simulated data        \n");  
    fprintf(stream , "-------------------------------------------------------------------------------------------------------------------------------\n");
    
    {
      int block_nr;
      int obs_count = 1;  /* Only for printing */
      for (block_nr =0; block_nr < obs_data_get_num_blocks( obs_data ); block_nr++) {
        const obs_block_type  * obs_block  = obs_data_iget_block_const( obs_data , block_nr);
        const meas_block_type * meas_block = meas_data_iget_block_const( meas_data , block_nr );
        const char * obs_key = obs_block_get_key( obs_block );
        
        for (int iobs = 0; iobs < obs_block_get_size( obs_block ); iobs++) {
          const char * print_key;
          if (iobs == 0)
            print_key = obs_key;
          else
            print_key = "  ...";
          
          fprintf(stream , obs_fmt , obs_count , print_key , obs_block_iget_value( obs_block , iobs ) , obs_block_iget_std( obs_block , iobs ));
          {
            active_type active_mode = obs_block_iget_active_mode( obs_block , iobs );
            if (active_mode == ACTIVE)
              fprintf(stream , "  Active   |");
            else if (active_mode == DEACTIVATED)
              fprintf(stream , "  Inactive |");
            else if (active_mode == MISSING)
              fprintf(stream , "           |");
            else
              util_abort("%s: enum_value:%d not handled - internal error\n" , __func__ , active_mode);
            if (active_mode == MISSING)
              fprintf(stream , "                  Missing\n");
            else
              fprintf(stream , sim_fmt, meas_block_iget_ens_mean( meas_block , iobs ) , meas_block_iget_ens_std( meas_block , iobs ));
          }
          obs_count++;
        }
      }
    }
    
    free( obs_fmt );
    free( sim_fmt );
  } 
  fprintf(stream , "===============================================================================================================================\n");
  fprintf(stream , "\n\n\n");
}




void enkf_analysis_deactivate_outliers(obs_data_type * obs_data , meas_data_type * meas_data , double std_cutoff , double alpha) {
  for (int block_nr =0; block_nr < obs_data_get_num_blocks( obs_data ); block_nr++) {
    obs_block_type  * obs_block  = obs_data_iget_block( obs_data , block_nr);
    meas_block_type * meas_block = meas_data_iget_block( meas_data , block_nr );
    
    meas_block_calculate_ens_stats( meas_block );
    {
      int iobs;
      for (iobs =0; iobs < meas_block_get_total_size( meas_block ); iobs++) {
        if (meas_block_iget_active( meas_block , iobs )) {
          double ens_std  = meas_block_iget_ens_std( meas_block , iobs );
          if (ens_std < std_cutoff) {
            /*
              De activated because the ensemble has to small
              variation for this particular measurement.
            */
            obs_block_deactivate( obs_block , iobs , "No ensemble variation");
            meas_block_deactivate( meas_block , iobs );
          } else {
            double ens_mean  = meas_block_iget_ens_mean( meas_block , iobs );
            double obs_std   = obs_block_iget_std( obs_block , iobs );
            double obs_value = obs_block_iget_value( obs_block , iobs );
            double innov     = obs_value - ens_mean;
            
            /* 
               Deactivated because the distance between the observed data
               and the ensemble prediction is to large. Keeping these
               outliers will lead to numerical problems.
            */

            if (fabs( innov ) > alpha * (ens_std + obs_std)) {
              obs_block_deactivate(obs_block , iobs , "No overlap");
              meas_block_deactivate(meas_block , iobs);
            }
          }
        }
      }
    }
  }
}



/*
  This function will allocate and initialize the matrices S,R,D and E
  and also the innovation. The matrices will be scaled with the
  observation error and the mean will be subtracted from the S matrix.
*/

static void enkf_analysis_alloc_matrices( rng_type * rng , 
                                          const meas_data_type * meas_data , obs_data_type * obs_data , enkf_mode_type enkf_mode , 
                                          matrix_type ** S , 
                                          matrix_type ** R , 
                                          double      ** innov,
                                          matrix_type ** E ,
                                          matrix_type ** D ) {
  int ens_size              = meas_data_get_ens_size( meas_data );
  int active_size           = obs_data_get_active_size( obs_data );

  *S                        = meas_data_allocS( meas_data , active_size );
  *R                        = obs_data_allocR( obs_data , active_size );
  *innov                    = obs_data_alloc_innov(obs_data , meas_data , active_size );
  
  if (enkf_mode == ENKF_STANDARD) {
    /* 
       We are using standard EnKF and need to perturbe the measurements,
       if we are using the SQRT scheme the E & D matrices are not used.
    */
    *E = obs_data_allocE(obs_data , rng , ens_size, active_size );
    *D = obs_data_allocD(obs_data , *E , *S );
    
  } else {
    *E          = NULL;
    *D          = NULL;
  }

  obs_data_scale(obs_data ,  *S , *E , *D , *R , *innov );
  matrix_subtract_row_mean( *S );  /* Subtracting the ensemble mean */
}

static void enkf_analysis_alloc_matrices_no_scaling( rng_type * rng , 
                                          const meas_data_type * meas_data , obs_data_type * obs_data , enkf_mode_type enkf_mode , 
                                          matrix_type ** S , 
                                          matrix_type ** R , 
                                          double      ** innov,
                                          matrix_type ** E ,
                                          matrix_type ** D ) {
  int ens_size              = meas_data_get_ens_size( meas_data );
  int active_size           = obs_data_get_active_size( obs_data );

  *S                        = meas_data_allocS( meas_data , active_size );
  *R                        = obs_data_allocR( obs_data , active_size );
  *innov                    = obs_data_alloc_innov(obs_data , meas_data , active_size );
  
  if (enkf_mode == ENKF_STANDARD) {
    /* 
       We are using standard EnKF and need to perturbe the measurements,
       if we are using the SQRT scheme the E & D matrices are not used.
    */
    *E = obs_data_allocE(obs_data , rng , ens_size, active_size );
    *D = obs_data_allocD(obs_data , *E , *S );
    
  } else {
    *E          = NULL;
    *D          = NULL;
  }

  /*obs_data_scale(obs_data ,  *S , *E , *D , *R , *innov );*/
  matrix_subtract_row_mean( *S );  /* Subtracting the ensemble mean */
}






static void enkf_analysis_alloc_matrices_boot( rng_type * rng , 
                                               const meas_data_type * meas_data , obs_data_type * obs_data , enkf_mode_type enkf_mode , 
                                               matrix_type ** S , 
                                               matrix_type ** R , 
                                               double      ** innov,
                                               matrix_type ** E ,
                                               matrix_type ** D ,
                                               const meas_data_type * fasit,
                                               matrix_type  ** fullS) {
  int ens_size              = meas_data_get_ens_size( meas_data );
  int active_size           = obs_data_get_active_size( obs_data );
  *S                        = meas_data_allocS( meas_data , active_size );
  *fullS                    = meas_data_allocS( fasit , active_size );
  *R                        = obs_data_allocR( obs_data , active_size );
  *innov                    = obs_data_alloc_innov(obs_data , meas_data , active_size );
  
  if (enkf_mode == ENKF_STANDARD) {
    /* 
       We are using standard EnKF and need to perturbe the measurements,
       if we are using the SQRT scheme the E & D matrices are not used.
    */
    *E = obs_data_allocE(obs_data , rng , ens_size, active_size );
    *D = obs_data_allocD(obs_data , *E , *fullS );
  } else {
    *E          = NULL;
    *D          = NULL;
  }

  obs_data_scale(obs_data ,  *S , *E , *D , *R , *innov );
  matrix_subtract_row_mean( *S );  /* Subtracting the ensemble mean */
}

/**
   Checking that the sum through one row in the X matrix is one.
*/

static void enkf_analysis_checkX(const matrix_type * X , bool bootstrap) {
  matrix_assert_finite( X );
  {
    int target_sum;
    if (bootstrap)
      target_sum = 0;
    else
      target_sum = 1;
    
    for (int icol = 0; icol < matrix_get_columns( X ); icol++) {
      double col_sum = matrix_get_column_sum(X , icol);
      if (fabs(col_sum - target_sum) > 0.0001) 
        util_abort("%s: something is seriously broken. col:%d  col_sum = %g != 1.0 - ABORTING\n",__func__ , icol , col_sum);
    }
  }
}




/**
   This function allocates a X matrix for the 

      A' = AX

   EnKF update. It takes as input a meas_data - where all the
   measurements have been collected, and a obs_data instance where the
   corresponding observations have been assembled. In addition it
   takes as input a random rotation matrix which will be used IFF the
   SQRT scheme is used, otherwise the random rotation matrix is not
   considered (and can be NULL).

   The function consists of three different parts:

   1.  The function starts with several function call allocating the
       ordinary matrices S, D, R, E and innov.

    2. The S matrix is 'diagonalized', and singular vectors and
       singular values are stored in W and eig.

    3. The actual X matrix is calculated, based on either the standard
       enkf with perturbed measurement or the square root scheme.

*/
   
matrix_type * enkf_analysis_allocX( const analysis_config_type * config , rng_type * rng , const meas_data_type * meas_data , obs_data_type * obs_data , const matrix_type * randrot) {
  int ens_size          = meas_data_get_ens_size( meas_data );
  matrix_type * X       = matrix_alloc( ens_size , ens_size );
  matrix_set_name( X , "X");
  {
    matrix_type * S , *R , *E , *D;
    double      * innov;
    int nrobs                = obs_data_get_active_size(obs_data);
    int nrmin                = util_int_min( ens_size , nrobs); 
    
    matrix_type * W          = matrix_alloc(nrobs , nrmin);                      
    double      * eig        = util_malloc( sizeof * eig * nrmin , __func__);    
    enkf_mode_type enkf_mode = analysis_config_get_enkf_mode( config );    
    bool bootstrap           = analysis_config_get_bootstrap( config );
    enkf_analysis_alloc_matrices( rng , meas_data , obs_data , enkf_mode , &S , &R , &innov , &E , &D );
        
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
      enkf_analysis_standard(X , S , D , W , eig , bootstrap);
      break;
    case(ENKF_SQRT):
      enkf_analysis_SQRT(X , S , randrot , innov , W , eig , bootstrap);
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

    enkf_analysis_checkX(X , bootstrap);
  }
  return X;
}


/*Research code for computing the update matrix: Includes the option of forcing the number of principal components: */
matrix_type * enkf_analysis_allocX_force( const analysis_config_type * config , rng_type * rng , const meas_data_type * meas_data , obs_data_type * obs_data , const matrix_type * randrot) {
  int ens_size          = meas_data_get_ens_size( meas_data );
  matrix_type * X       = matrix_alloc( ens_size , ens_size );
  matrix_set_name( X , "X");
  {
    matrix_type * S , *R , *E , *D;
    double      * innov;
    int nrobs                = obs_data_get_active_size(obs_data);
    int nrmin                = util_int_min( ens_size , nrobs); 
    
    /*retrieve the subspace dimension we want to use */
    int ncomp                = analysis_config_get_subspace_dimension( config );

    /* Check if we want to run a bootrap */
    bool bootstrap           = analysis_config_get_bootstrap( config );
    
    /*Check if we want to scale the data or not */
    bool do_scaling          = analysis_config_get_do_scaling( config );
    /*HERE WE ARE*/
    

    /* Check if the dimension is appropropriate. If not, change to default */
    if (ncomp > util_int_min( ens_size - 1, nrobs) ) {
      printf("Selected number of components, %d, too high. Changing to default value: 1\n",ncomp);
      ncomp = 1;
    }
    
	
    
    matrix_type * W          = matrix_alloc(nrobs , nrmin);                      
    double      * eig        = util_malloc( sizeof * eig * nrmin , __func__);    
    enkf_mode_type enkf_mode = analysis_config_get_enkf_mode( config );    
    if (do_scaling) {
      enkf_analysis_alloc_matrices( rng , meas_data , obs_data , enkf_mode , &S , &R , &innov , &E , &D );
    }
    else {
      printf("\nWarning: Scaling of forecasted data turned off! Generally this is not recommended...\n");
      enkf_analysis_alloc_matrices_no_scaling( rng , meas_data , obs_data , enkf_mode , &S , &R , &innov , &E , &D );
    }
        
    /* 
       2: Diagonalize the S matrix; singular vectors are stored in W
          and singular values (after some massage) are stored in eig. 
          W = X1, eig = inv(I+Lambda1),(Eq.14.30, and 14.29, Evensen, 2007, respectively)
    */ 

    enkf_analysis_invertS_force( config , S , R , W , eig , ncomp);
    
    /* 
       3: actually calculating the X matrix. 
    */
    switch (enkf_mode) {
    case(ENKF_STANDARD):
      enkf_analysis_standard(X , S , D , W , eig , bootstrap);
      break;
    case(ENKF_SQRT):
      enkf_analysis_SQRT(X , S , randrot , innov , W , eig , bootstrap);
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

    enkf_analysis_checkX(X , bootstrap);
  }
  return X;
}






/**
   This function allocates a X matrix for the 

      A' = AX

   EnKF update. Same as above except we do not want a resampled version of the D-matrix so
   for bootstrapping purposes we also need the unsampled meas_data.

*/

matrix_type * enkf_analysis_allocX_boot( const analysis_config_type * config , rng_type * rng , const meas_data_type * meas_data , obs_data_type * obs_data , const matrix_type * randrot , const meas_data_type * fasit) {
  int ens_size          = meas_data_get_ens_size( meas_data );
  matrix_type * X       = matrix_alloc( ens_size , ens_size );
  matrix_set_name( X , "X");
  {
    matrix_type * S , *R , *E , *D;
    matrix_type * fullS;
    double      * innov;
    int nrobs                = obs_data_get_active_size(obs_data);
    int nrmin                = util_int_min( ens_size , nrobs); 
    
    matrix_type * W          = matrix_alloc(nrobs , nrmin);                      
    double      * eig        = util_malloc( sizeof * eig * nrmin , __func__);    
    enkf_mode_type enkf_mode = analysis_config_get_enkf_mode( config );    
    bool bootstrap           = analysis_config_get_bootstrap( config );
    enkf_analysis_alloc_matrices_boot( rng , meas_data , obs_data , enkf_mode , &S , &R , &innov , &E , &D , fasit , &fullS);
    
    matrix_free( fullS );
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
      enkf_analysis_standard(X , S , D , W , eig , bootstrap);
      break;
    case(ENKF_SQRT):
      enkf_analysis_SQRT(X , S , randrot , innov , W , eig , bootstrap);
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

    enkf_analysis_checkX(X , bootstrap);
  }
  return X;
}


matrix_type * enkf_analysis_allocX_pre_cv( const analysis_config_type * config , rng_type * rng , meas_data_type * meas_data , obs_data_type * obs_data , 
					   const matrix_type * randrot , const matrix_type * A , const matrix_type * V0T , const matrix_type * Z ,
					   const double * eig , const matrix_type * U0 , meas_data_type * fasit , int unique_bootstrap_components) {
  int ens_size          = meas_data_get_ens_size( meas_data );
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
    matrix_type * fullS;
    double      * innov;
    
    matrix_type * W          = matrix_alloc(nrobs , nrmin);                      
    enkf_mode_type enkf_mode = analysis_config_get_enkf_mode( config );    
    bool bootstrap           = analysis_config_get_bootstrap( config );    
    bool penalised_press     = analysis_config_get_penalised_press( config );
    
    double * workeig    = util_malloc( sizeof * workeig * nrmin , __func__);

    enkf_analysis_alloc_matrices_boot( rng , meas_data , obs_data , enkf_mode , &S , &R , &innov , &E , &D , fasit , &fullS ); /*Using the bootstrap version every time, does mean a bit more data 
                                                                                                                           carried through the function, but we avoid duplicating code.*/
    matrix_free( fullS );

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

    getW_pre_cv(W , V0T , Z , workeig ,  U0 , nfolds_CV , workA , unique_bootstrap_components , rng , penalised_press);
    
    /* 
       3: actually calculating the X matrix. 
    */
    switch (enkf_mode) {
    case(ENKF_STANDARD):
      enkf_analysis_standard(X , S , D , W , workeig , bootstrap);
      break;
    case(ENKF_SQRT):
      enkf_analysis_SQRT(X , S , randrot , innov , W , workeig , bootstrap );
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
    
    enkf_analysis_checkX(X , bootstrap);
  }
  return X;
}





/** 
    This function initializes the S matrix and performs svd(S). The
    left and right singular vectors and singular values are returned
    in U0, V0T and eig respectively.
*/

void enkf_analysis_local_pre_cv( const analysis_config_type * config , rng_type * rng , meas_data_type * meas_data , obs_data_type * obs_data ,  matrix_type * V0T , matrix_type * Z , double * eig , matrix_type * U0, meas_data_type * fasit ) {
  {
    matrix_type * S , *R , *E , *D;
    matrix_type * fullS;
    double      * innov;
    
    enkf_mode_type enkf_mode = analysis_config_get_enkf_mode( config );
    enkf_analysis_alloc_matrices_boot( rng , meas_data , obs_data , enkf_mode , &S , &R , &innov , &E , &D , fasit , &fullS ); /*Using the bootstrap version every time, does mean a bit more data 
                                                                                                                           carried through the function, but we avoid duplicating code.*/
    matrix_free( fullS );
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


/**
  FUNCTION THAT COMPUTES THE PRINCIPAL COMPONENTS OF THE CENTRED DATA ENSEMBLE MATRIX:
  

  This function initializes the S matrix and performs svd(S). The
  Function returns:
  Z  -   The Principal Components of the empirically estimated data covariance matrix as Z (ens_size times maxP), where ens_size is the 
             ensemble size, and maxP is the maximum number of allowed principal components
  Rp -   (Rp = U0' * R * U0 (maxP times maxP) error covariance matrix in the reduced order subspace (Needed later in the EnKF update)
  
  Dp -   (Dp = U0' * D) (maxP times ens_size): Reduced data "innovation matrix".
         where D(:,i) = dObs - dForecast(i) - Eps(i)
*/


void enkf_analysis_get_principal_components( const analysis_config_type * config , rng_type * rng , meas_data_type * meas_data , obs_data_type * obs_data ,  matrix_type * Z , matrix_type * Rp , matrix_type * Dp) {
  {
    matrix_type * S , *R , *E , *D;
    double      * innov;
    int i, j;

    printf("\nInside the Principal Components Function\n");
    
    enkf_mode_type enkf_mode = analysis_config_get_enkf_mode( config );
    enkf_analysis_alloc_matrices( rng , meas_data , obs_data , enkf_mode , &S , &R , &innov , &E , &D ); 


    const int nrobs = matrix_get_rows( S );
    const int nrens = matrix_get_columns( S );
    const int nrmin = util_int_min( nrobs , nrens );

    printf("Maximum number of Principal Components is %d\n",nrmin);

    double truncation                     = analysis_config_get_truncation( config );  
    

    /*
      Compute SVD(S)
    */
      
    matrix_type * U0   = matrix_alloc( nrobs , nrmin    ); /* Left singular vectors.  */
    matrix_type * V0T  = matrix_alloc( nrmin , nrens ); /* Right singular vectors. */
    
    double * sig0      = util_malloc( nrmin * sizeof * sig0 , __func__);
    
    svdS(S , U0 , V0T , DGESVD_MIN_RETURN , sig0 , truncation);

    /* NEED TO invert sig0!!  */
    for(i = 0; i < nrmin; i++) {
      if ( sig0[i] > 0 ) {
	sig0[i] = 1 / sig0[i];
      }
    }
	  
    
    /*
	Compute the actual princpal components, Z = sig0 * VOT 
	NOTE: Z contains potentially alot of redundant zeros, but 
	we do not care about this for now
	      
    */
    for(i = 0; i < nrmin; i++) {
      for(j = 0; j < nrens; j++) {
	matrix_iset( Z , i , j , sig0[i] * matrix_iget( V0T , i , j ) );
      }
    }
    
    /* Also compute Rp */

    matrix_type * X0 = matrix_alloc( nrmin , matrix_get_rows( R ));
    matrix_dgemm(X0 , U0 , R  , true  , false , 1.0 , 0.0);  /* X0 = U0^T * R */
    matrix_dgemm(Rp  , X0 , U0 , false , false , 1.0 , 0.0);  /* Rp = X0 * U0 */
    matrix_free( X0 );

    /*We also need to compute the reduced "Innovation matrix" Dp = U0' * D    */
    matrix_dgemm(Dp , U0 , D , true , false , 1.0 , 0.0);
    
	
    free(sig0);
    matrix_free(U0);
    matrix_free(V0T);
    
    printf("Returning the Pre-Computed Principal Components\n");
    printf("\n");

    /* 
       2: Diagonalize the S matrix; singular vectors etc. needed later in the local CV:
       (V0T = transposed right singular vectors of S, Z = scaled principal components, 
       eig = scaled, inverted singular vectors, U0 = left singular vectors of S
       , eig = inv(I+Lambda1),(Eq.14.30, and 14.29, Evensen, 2007, respectively)
    */ 
    /*   enkf_analysis_invertS_pre_cv( config , S , R , V0T , Z , eig , U0);*/
    
    
    matrix_free( R );
    matrix_free( S );
    free( innov );
        
    if (enkf_mode == ENKF_STANDARD) {
      matrix_free( E );
      matrix_free( D );
      
    }
  }
  
}


/*Matrix that computes and returns the X5 matrix used in the EnKF updating */
matrix_type * enkf_analysis_allocX_principal_components_cv( const analysis_config_type * config , rng_type * rng, const matrix_type * A , const matrix_type * Z , const matrix_type * Rp , const matrix_type * Dp) {
  int ens_size = matrix_get_columns( Dp );
  matrix_type * X       = matrix_alloc( ens_size , ens_size );
  {

    int nfolds_CV            = analysis_config_get_nfolds_CV( config );
    bool bootstrap           = analysis_config_get_bootstrap( config );
    bool penalised_press     = analysis_config_get_penalised_press( config );
    int i, j, k;
    double tmp;
      

    /*
      1: Allocating all matrices
    */
    /*Need a copy of A, because we need it later */
    matrix_type * workA      = matrix_alloc_copy( A );    /* <- This is a massive memory requirement. */
    
    /* Subtracting the ensemble mean of the state vector ensemble */
    matrix_subtract_row_mean( workA );
    /* 
       2: Diagonalize the S matrix; singular vectors are stored in W
          and singular values (after some massage) are stored in eig. 
    W = X1, eig = inv(I+Lambda1),(Eq.14.30, and 14.29, Evensen, 2007, respectively)
    */

    
    
    int nrmin = matrix_get_rows( Z );
    int maxP = nrmin;

    /* We only want to search the non-zero eigenvalues */
    for (int i = 0; i < nrmin; i++) {
      if (matrix_iget(Z,i,1) == 0.0) {
	maxP = i;
	break;
      }
    }
    
    if (maxP > nrmin) {
      maxP = nrmin;
    }



    
    /* Get the optimal number of principal components 
       where p is found minimizing the PRESS statistic */
    
    int optP = get_optimal_principal_components(Z , Rp , nfolds_CV, workA , rng , maxP, penalised_press);
    matrix_free( workA );

    matrix_type * W          = matrix_alloc(ens_size , optP);                      

    /* Compute  W = Z(1:p,:)' * inv(Z(1:p,:) * Z(1:p,:)' + (ens_size-1) * Rp(1:p,1:p))*/
    getW_prin_comp( W , Z , Rp, optP);

    /*Compute the actual X5 matrix: */
    /*Compute X5 = W * Dp (The hard way) */
    for( i = 0; i < ens_size; i++) {
      for( j = 0; j < ens_size; j++) {
	tmp = 0.0;
	for(k = 0; k < optP; k++) {
	  tmp += matrix_iget( W , i , k) * matrix_iget( Dp , k , j);
	}
	
	matrix_iset(X , i , j ,tmp);
      }
    }

    matrix_free( W );
    
    /*Add one on the diagonal of X: */
    for(i = 0; i < ens_size; i++) {
      matrix_iadd( X , i , i , 1.0); /*X5 = I + X5 */
    }
    
    enkf_analysis_checkX(X , bootstrap);
  }
  

  
  return X;
}










