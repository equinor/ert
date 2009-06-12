#include <util.h>
#include <math.h>
#include <matrix.h>
#include <matrix_lapack.h>
#include <matrix_blas.h>
#include <meas_matrix.h>
#include <obs_data.h>
#include <analysis_config.h>

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



static void lowrankCinv(matrix_type * S , matrix_type * R , matrix_type * W , double * eig , double truncation) {
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


/*****************************************************************/
/*****************************************************************/
/*                     High level functions                      */
/*****************************************************************/
/*****************************************************************/


/**
   Observe the following about the S matrix:

   1. On input the matrix is supposed to contain actual measurement values.
   2. On return the ensemble mean has been shifted away ...
   
*/


static void enkf_analysis_standard_lowrankCinv(matrix_type * X5 , matrix_type * R , matrix_type * S , const matrix_type * D , double truncation) {
  const int nrobs   = matrix_get_rows( S );
  const int nrens   = matrix_get_columns( S );
  const int nrmin   = util_int_min( nrobs , nrens );
  
  matrix_type * X3    = matrix_alloc(nrobs , nrens);
  matrix_type * W     = matrix_alloc(nrobs , nrmin);
  double      * eig   = util_malloc( sizeof * eig * nrmin , __func__);
  matrix_subtract_row_mean( S );  /* Subtracting the ensemble mean */
  {
    /* 
       The svd routine called in lowrankCinv will destroy the contents
       of the input matrix, we therefor have to store a copy of S
       befor calling it. The fortran implementation seems to handle
       this automagically??
    */
    matrix_type * workS = matrix_alloc_copy( S );  
    lowrankCinv( workS , R , W , eig , truncation );
    matrix_free( workS );
  }
  
  genX3(X3 , W , D , eig );     
  matrix_dgemm( X5 , S , X3 , true , false , 1.0 , 0.0);  /* X5 = T(S) * X3 */
  {
    for (int i = 0; i < nrens; i++)
      matrix_iadd( X5 , i , i , 1.0);
  }
  
  free( eig );
  matrix_free( W );
  matrix_free( X3 );
}



/*****************************************************************/


void enkf_analysis_fprintf_obs_summary(const obs_data_type * obs_data , const meas_matrix_type * meas_matrix , FILE * stream ) {
  int iobs;
  fprintf(stream , "/-----------------------------------------------------------------------|---------------------------------\\\n");
  fprintf(stream , "|                          Observed history                             |         Simulated data          |\n");  
  fprintf(stream , "|-----------------------------------------------------------------------|---------------------------------|\n");
  for (iobs = 0; iobs < obs_data_get_nrobs(obs_data); iobs++) {
    obs_data_node_type * node = obs_data_iget_node( obs_data , iobs);
    

    fprintf(stream , "| %-3d : %-16s    %12.3f +/-  %12.3f ",iobs + 1 , 
	    obs_data_node_get_keyword(node),
	    obs_data_node_get_value(node),
	    obs_data_node_get_std(node));
    
    if (obs_data_node_active( node )) 
      fprintf(stream , "   Active    |");
    else
      fprintf(stream , "   Inactive  |");
    {
      double mean,std;
      meas_matrix_iget_ens_mean_std( meas_matrix , iobs , &mean , &std);
      fprintf(stream , "   %12.3f +/- %12.3f |\n", mean , std);
    }
  }
  fprintf(stream , "\\-----------------------------------------------------------------------|---------------------------------/\n");
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





matrix_type * enkf_analysis_allocX( const analysis_config_type * config , meas_matrix_type * meas_matrix , obs_data_type * obs_data ) {
  double truncation = 0.95;
  
  {
    enkf_mode_type        enkf_mode       = analysis_config_get_enkf_mode( config );
    pseudo_inversion_type inversion_mode  = analysis_config_get_inversion_mode( config );
    
    int ens_size    = meas_matrix_get_ens_size( meas_matrix );
    matrix_type * X = matrix_alloc( ens_size , ens_size );
    matrix_type * S = meas_matrix_allocS__( meas_matrix );
    matrix_type * R = obs_data_allocR__(obs_data);
    matrix_type * E = obs_data_allocE__(obs_data , ens_size);
    matrix_type * D = obs_data_allocD__(obs_data , E , S);
    
    obs_data_scale__(obs_data , S , E , D , R, NULL /* innov */);

    if (enkf_mode == ENKF_STANDARD) {
      switch (inversion_mode) {
      case(SVD_SS_N1_R):
	enkf_analysis_standard_lowrankCinv(X , R , S , D , truncation );
	break;
      default:
	util_abort("%s: inversion mode:%d not supported \n",__func__ , inversion_mode);
      }
    } else if (enkf_mode == ENKF_SQRT) {
      util_abort("%s: enkf mode:%d not supported \n",__func__ , enkf_mode);
    } else
      util_abort("%s: INTERNAL ERROR \n",__func__);
    
    matrix_free( R );
    matrix_free( E );
    matrix_free( S );
    matrix_free( D );

    return X;
  }
}
