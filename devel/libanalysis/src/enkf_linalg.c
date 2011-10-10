#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <matrix.h>
#include <matrix_lapack.h>
#include <matrix_blas.h>
#include <util.h>

#include <enkf_linalg.h>

void enkf_linalg_genX3(matrix_type * X3 , const matrix_type * W , const matrix_type * D , const double * eig) {
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


void enkf_linalg_genX2(matrix_type * X2 , const matrix_type * S , const matrix_type * W , const double * eig) {
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
   This function calculates the svd of the input matrix S. The number
   of significant singular values to retain can either be forced to a
   fixed number, or a cutoff on small singular values can be
   used. This behaviour is regulated by the @ncomp and @truncation
   parameters:

     ncomp > 0 , truncation < 0: Use ncomp parameters.
     ncomp < 0 , truncation > 0: Truncate at level 'truncation'.
     
   The singular values are returned in the inv_sig0 vector; the values
   we retain are inverted and the remaining elements in are explicitly
   set to zero.
   
   The left-hand singular vectors are returned in the matrix
   U0. Depending on the value of the flag @store_V0T the right hand
   singular vectors are stored in the V0T matrix, or just
   discarded. If you do not intend to use the right hand vectors at
   all, i.e. store_V0T == DGESVD_NONE, the V0T matrix will not be
   accessed.
*/



void enkf_linalg_svdS(const matrix_type * S , 
                      double truncation , 
                      int ncomp ,
                      dgesvd_vector_enum store_V0T , 
                      double * inv_sig0, 
                      matrix_type * U0 , 
                      matrix_type * V0T) {
  
  double * sig0 = inv_sig0;
  
  if (((truncation > 0) && (ncomp < 0)) ||
      ((truncation < 0) && (ncomp > 0))) {
    int num_singular_values = util_int_min( matrix_get_rows( S ) , matrix_get_columns( S ));
    {
      /* 
         The svd routine will destroy the contents of the input matrix,
         we therefor have to store a copy of S before calling it. 
      */
      matrix_type * workS = matrix_alloc_copy( S );
      matrix_dgesvd(DGESVD_MIN_RETURN , store_V0T , workS , sig0 , U0 , V0T);  
      matrix_free( workS );
    }
    
    {
      int    num_significant;
      int i;
      
      if (ncomp > 0)
        num_significant = ncomp;
      else {
        double total_sigma2    = 0;
        for (i=0; i < num_singular_values; i++)
          total_sigma2 += sig0[i] * sig0[i];
        
        /* 
           Determine the number of singular values by enforcing that
           less than a fraction @truncation of the total variance be
           accounted for. 
        */
        num_significant = 0;
        {
          double running_sigma2  = 0;
          for (i=0; i < num_singular_values; i++) {
            if (running_sigma2 / total_sigma2 < truncation) {  /* Include one more singular value ? */
              num_significant++;
              running_sigma2 += sig0[i] * sig0[i];
            } else 
              break;
          }
        }
      }
      
      /* Inverting the significant singular values */
      for (i = 0; i < num_significant; i++)
        inv_sig0[i] = 1.0 / sig0[i];

      /* Explicitly setting the insignificant singular values to zero. */
      for (i=num_significant; i < num_singular_values; i++)
        inv_sig0[i] = 0;                                     
    }
  } else 
    util_abort("%s:  truncation:%g  ncomp:%d  - invalid ambigous input.\n",__func__ , truncation , ncomp );
}




static void lowrankCee(matrix_type * B, int nrens , const matrix_type * R , const matrix_type * U0 , const double * inv_sig0) {
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
        matrix_imul(B , i , j , inv_sig0[i]);

    for (j=0; j < matrix_get_columns( B ) ; j++)
      for (i=0; i < matrix_get_rows( B ); i++)
        matrix_imul(B , i , j , inv_sig0[j]);
  }
  
  matrix_scale(B , nrens - 1.0);
}




void enkf_linalg_lowrankCinv__(const matrix_type * S , 
                               const matrix_type * R , 
                               matrix_type * V0T , 
                               matrix_type * Z, 
                               double * eig , 
                               matrix_type * U0, 
                               double truncation, 
                               int ncomp) {
  
  const int nrobs = matrix_get_rows( S );
  const int nrens = matrix_get_columns( S );
  const int nrmin = util_int_min( nrobs , nrens );
  
  double * inv_sig0      = util_malloc( nrmin * sizeof * inv_sig0 , __func__);

  if (V0T != NULL)
    enkf_linalg_svdS(S , truncation , ncomp , DGESVD_MIN_RETURN , inv_sig0 , U0 , V0T );
  else
    enkf_linalg_svdS(S , truncation , ncomp , DGESVD_NONE , inv_sig0, U0 , NULL);

  {
    matrix_type * B    = matrix_alloc( nrmin , nrmin );
    lowrankCee( B , nrens , R , U0 , inv_sig0);          /* B = Xo = (N-1) * Sigma0^(+) * U0'* Cee * U0 * Sigma0^(+')  (14.26)*/     
    matrix_dgesvd(DGESVD_MIN_RETURN , DGESVD_NONE, B , eig, Z , NULL);
    matrix_free( B );
  }
  
  {
    int i,j;
    /* Lambda1 = (I + Lambda)^(-1) */

    for (i=0; i < nrmin; i++) 
      eig[i] = 1.0 / (1 + eig[i]);
    
    for (j=0; j < nrmin; j++)
      for (i=0; i < nrmin; i++)
        matrix_imul(Z , i , j , inv_sig0[i]); /* Z2 =  Sigma0^(+) * Z; */
  }
  util_safe_free( inv_sig0 );
}


void enkf_linalg_lowrankCinv(const matrix_type * S , 
                             const matrix_type * R , 
                             matrix_type * W       , /* Corresponding to X1 from Eq. 14.29 */
                             double * eig          , /* Corresponding to 1 / (1 + Lambda_1) (14.29) */
                             double truncation     ,
                             int    ncomp) {
  
  const int nrobs = matrix_get_rows( S );
  const int nrens = matrix_get_columns( S );
  const int nrmin = util_int_min( nrobs , nrens );

  matrix_type * U0   = matrix_alloc( nrobs , nrmin );
  matrix_type * Z    = matrix_alloc( nrmin , nrmin );
  
  enkf_linalg_lowrankCinv__( S , R , NULL , Z , eig , U0 , truncation , ncomp);
  matrix_matmul(W , U0 , Z); /* X1 = W = U0 * Z2 = U0 * Sigma0^(+') * Z    */

  matrix_free( U0 );
  matrix_free( Z  );
}


void enkf_linalg_meanX5(const matrix_type * S , 
                        const matrix_type * W , 
                        const double * eig    , 
                        const matrix_type * dObs, 
                        matrix_type * X5) {

  
  const int nrens = matrix_get_columns( S );
  const int nrobs = matrix_get_rows( S );
  const int nrmin = util_int_min( nrobs , nrens );
  double * work   = util_malloc( (2 * nrmin + nrobs + nrens) * sizeof * work , __func__);
  matrix_type * innov = enkf_linalg_alloc_innov( dObs , S );
  {
    double * y1 = &work[0];
    double * y2 = &work[nrmin];
    double * y3 = &work[2*nrmin];
    double * y4 = &work[2*nrmin + nrobs]; 
    
    if (nrobs == 1) {
      /* Is this special casing necessary ??? */
      y1[0] = matrix_iget(W , 0,0) * matrix_iget( innov , 0 , 0);
      y2[0] = eig[0] * y1[0];
      y3[0] = matrix_iget(W , 0, 0) *y2[0];
      for (int iens = 0; iens < nrens; iens++)
        y4[iens] = y3[0] * matrix_iget(S , 0, iens);
    } else {
      matrix_dgemv(W , matrix_get_data( innov ) , y1 , true , 1.0, 0.0);   /* y1 = Trans(W) * innov */
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
  matrix_free( innov );
}



void enkf_linalg_X5sqrt(matrix_type * X2 , matrix_type * X5 , const matrix_type * randrot, int nrobs) { 
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


matrix_type * enkf_linalg_alloc_innov( const matrix_type * dObs , const matrix_type * S) {
  matrix_type * innov = matrix_alloc_copy( dObs );
  for (int iobs =0; iobs < matrix_get_row_sum( dObs , iobs); iobs++) 
    matrix_isub( innov , iobs , 0 , matrix_get_row_sum( S , iobs ));
}
