#include <util.h>
#include <math.h>
#include <matrix.h>
#include <matrix_lapack.h>
#include <matrix_blas.h>

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
  matrix_matmul(X3 , W  , X2);   /* <- skal bare ha en del av X2 ??? */
  
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


static void svdS(matrix_type * S , matrix_type * U0 , double * sig0, double truncation) {
  int num_singular_values = util_int_min( matrix_get_rows( S ) , matrix_get_columns( S ));
  matrix_dgesvd(DGESVD_MIN_RETURN , DGESVD_NONE , S , sig0 , U0 , NULL);                   /* Have singular values in s0, and left hand singular vectors in U0 */
  {
    double total_sigma2    = 0;
    double running_sigma2  = 0;
    int    num_significant = 0;
    int i;
    for (i=0; i < num_singular_values; i++)
      total_sigma2 += sig0[i] * sig0[i];

    for (i=0; i < num_singular_values; i++) {
      if (running_sigma2 / total_sigma2 < truncation) {  /* Include one more singular value */
	num_significant++;
	running_sigma2 += sig0[i] * sig0[i];
      } else 
	break;
    }

    /* Explicitly setting the insignificant singular values to zero. */
    for (i=num_significant; i < num_singular_values; i++)
      sig0[0] = 0;                                     
    
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



static void lowrankCinv(matrix_type * S , matrix_type * R , matrix_type * W , double * eig , double truncation) {
  const int nrobs = matrix_get_rows( S );
  const int nrens = matrix_get_columns( S );
  const int nrmin = util_int_min( nrobs , nrens );
  
  matrix_type * B  = matrix_alloc( nrmin , nrmin );
  matrix_type * U0 = matrix_alloc( nrobs , nrmin );
  matrix_type * Z  = matrix_alloc( nrmin , nrmin );
  double * sig0    = util_malloc( nrmin * sizeof * sig0 , __func__);
    
  svdS(S , U0 , sig0 , truncation);
  lowrankCee( B , nrens , R , U0 , sig0);
  eigC( R , Z , eig );
  
  
  {
    int i,j;
    for (i=0; i < nrmin; i++) 
      eig[i] = 1.0 / (1 + eig[i]);

    for (j=0; j < nrmin; j++)
      for (i=0; i < nrmin; i++)
	matrix_imul(Z , i , j , sig0[i]);
  }
  
  
  free( sig0 );
  matrix_free( U0 );
  matrix_free( B  );
  matrix_free( Z  );
}


/*****************************************************************/
/*****************************************************************/
/*                     High level functions                      */
/*****************************************************************/
/*****************************************************************/



void enkf_analysis_standard_lowrankCinv(matrix_type * X5 , matrix_type * R , const matrix_type * E , matrix_type * S , const matrix_type * D , double truncation) {
  const int nrobs   = matrix_get_rows( S );
  const int nrens   = matrix_get_columns( S );
  const int nrmin   = util_int_min( nrobs , nrens );
  
  matrix_type * X3  = matrix_alloc(nrobs , nrens);
  matrix_type * W   = matrix_alloc(nrobs , nrmin);
  double      * eig = util_malloc( sizeof * eig * nrmin , __func__);
  
  
  lowrankCinv( S , R , W , eig , truncation );
  genX3(X3 , W , D , eig );
  matrix_dgemm( X5 , S , X3 , true , false , 1.0 , 0.0);  /* X5 = T(S) * X3 */
  {
    int i;
    for (i = 0; i < nrens; i++)
      matrix_iadd( X5 , i ,i , 1.0);
  }
  
  free( eig );
  matrix_free( W );
  matrix_free( X3 );
}

