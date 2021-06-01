/*
   Copyright (C) 2019  Equinor ASA, Norway.

   The file 'ies_enkf.c' is part of ERT - Ensemble based Reservoir Tool.

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


#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <ert/util/util.hpp>
#include <ert/util/rng.hpp>
#include <ert/util/bool_vector.hpp>

#include <ert/res_util/matrix.hpp>
#include <ert/res_util/matrix_blas.hpp>


#include <ert/analysis/analysis_module.hpp>
#include <ert/analysis/analysis_table.hpp>
#include <ert/analysis/enkf_linalg.hpp>

#include "ies_enkf_config.h"
#include "ies_enkf_data.h"


#ifdef __cplusplus
extern "C" {
#endif


/***************************************************************************************************************/
/* ies_enkf function headers */
/***************************************************************************************************************/
void ies_enkf_linalg_extract_active(const ies_enkf_data_type * data,
                                    matrix_type * E,
                                    FILE * log_fp,
                                    bool dbg);

void ies_enkf_linalg_compute_AA_projection(const matrix_type * A,
                                                 matrix_type * Y,
                                                 FILE * log_fp,
                                                 bool dbg);

void ies_enkf_linalg_extract_active_W(const ies_enkf_data_type * data,
                                            matrix_type * W0,
                                            FILE * log_fp,
                                            bool dbg);

void ies_linalg_solve_S(const matrix_type * W0,
                        const matrix_type * Y,
                              matrix_type * S,
                              double rcond,
                              FILE * log_fp,
                              bool dbg);

void ies_enkf_linalg_subspace_inversion(matrix_type * W0,
                                        const int ies_inversion,
                                        matrix_type * E,
                                        matrix_type * R,
                                        const matrix_type * S,
                                        const matrix_type * H,
                                        double truncation,
                                        double ies_steplength,
                                        int subspace_dimension,
                                        FILE * log_fp,
                                        bool dbg);

void ies_enkf_linalg_exact_inversion(matrix_type * W0,
                                     const int ies_inversion,
                                     const matrix_type * S,
                                     const matrix_type * H,
                                     double ies_steplength,
                                     FILE * log_fp,
                                     bool dbg);

void ies_enkf_linalg_store_active_W(ies_enkf_data_type * data,
                                    const matrix_type * W0,
                                    FILE * log_fp,
                                    bool dbg);

void ies_enkf_linalg_extract_active_A0(const ies_enkf_data_type * data,
                                       matrix_type * A0,
                                       FILE * log_fp,
                                       bool dbg);
/***************************************************************************************************************/

#ifdef __cplusplus
}
#endif


#define ENKF_SUBSPACE_DIMENSION_KEY      "ENKF_SUBSPACE_DIMENSION"
#define ENKF_TRUNCATION_KEY              "ENKF_TRUNCATION"
#define IES_MAX_STEPLENGTH_KEY           "IES_MAX_STEPLENGTH"
#define IES_MIN_STEPLENGTH_KEY           "IES_MIN_STEPLENGTH"
#define IES_DEC_STEPLENGTH_KEY           "IES_DEC_STEPLENGTH"
#define ITER_KEY                         "ITER"

#define IES_SUBSPACE_KEY                 "IES_SUBSPACE"
#define IES_INVERSION_KEY                "IES_INVERSION"
#define IES_LOGFILE_KEY                  "IES_LOGFILE"
#define IES_DEBUG_KEY                    "IES_DEBUG"
#define IES_AAPROJECTION_KEY             "IES_AAPROJECTION"


#include "tecplot.c"

//#define DEFAULT_ANALYSIS_SCALE_DATA true


static void printf_mask(FILE * log_fp, const char * name, const bool_vector_type * mask) {
  fprintf(log_fp,"%10s: ",name);
  for (int i = 0; i < bool_vector_size(mask); i++){
    fprintf(log_fp,"%d", bool_vector_iget(mask, i));
    if ((i+1)%10  == 0)  fprintf(log_fp,"%s"," ");
    if ((i+1)%100 == 0)  fprintf(log_fp,"\n %9d  " ,i+1);
  }
  fputs("\n", log_fp);
}


/***************************************************************************************************************
*  Set / Get iteration number
****************************************************************************************************************/
/*
 * -------------------------------------------------------------------------------------------------------------
 * I E n K S
 * IEnKS initialization (getting the obs_mask and ens_mask for active observations and realizations)
 * -------------------------------------------------------------------------------------------------------------
*/
void ies_enkf_init_update(void * arg ,
                          const bool_vector_type * ens_mask ,
                          const bool_vector_type * obs_mask ,
                          const matrix_type * S ,
                          const matrix_type * R ,
                          const matrix_type * dObs ,
                          const matrix_type * E ,
                          const matrix_type * D,
                          rng_type * rng) {
  ies_enkf_data_type * module_data = ies_enkf_data_safe_cast( arg );

/* Store current ens_mask in module_data->ens_mask for each iteration */
  ies_enkf_data_update_ens_mask(module_data, ens_mask);

/* Store obs_mask for initial iteration in module_data->obs_mask0,
*  for each subsequent iteration we store the current mask in module_data->obs_mask */
  ies_enkf_store_initial_obs_mask(module_data, obs_mask);
  ies_enkf_update_obs_mask(module_data, obs_mask);
}



/*
 * -------------------------------------------------------------------------------------------------------------
 * I E n K S
 * IEnKS update (IES searching the solution in ensemble subspace)
 * -------------------------------------------------------------------------------------------------------------
*/

void ies_enkf_updateA( void * module_data,
                       matrix_type * A ,      // Updated ensemble A retured to ERT.
                       const matrix_type * Yin ,    // Ensemble of predicted measurements
                       const matrix_type * Rin ,    // Measurement error covariance matrix (not used)
                       const matrix_type * dObs ,   // Actual observations (not used)
                       const matrix_type * Ein ,    // Ensemble of observation perturbations
                       const matrix_type * Din ,    // (d+E-Y) Ensemble of perturbed observations - Y
                       const module_info_type * module_info,
                       rng_type * rng) {

   ies_enkf_data_type * data = ies_enkf_data_safe_cast( module_data );
   const ies_enkf_config_type * ies_config = ies_enkf_data_get_config( data );


   int nrobs_msk     =ies_enkf_data_get_obs_mask_size(data); // Total number of observations
   int nrobs_inp     =matrix_get_rows( Yin );                // Number of active observations input in current iteration


   int ens_size_msk  = ies_enkf_data_get_ens_mask_size(data);   // Total number of realizations
   int ens_size      = matrix_get_columns( Yin );               // Number of active realizations in current iteration
   int state_size    = matrix_get_rows( A );

   ies_inversion_type ies_inversion = ies_enkf_config_get_ies_inversion( ies_config );
   double truncation = ies_enkf_config_get_truncation( ies_config );
   bool ies_debug = ies_enkf_config_get_ies_debug(ies_config);
   int subspace_dimension = ies_enkf_config_get_enkf_subspace_dimension( ies_config );
   double rcond=0;
   int nrsing=0;
   int iteration_nr = ies_enkf_data_inc_iteration_nr(data);
   FILE * log_fp;

   ies_enkf_data_update_state_size( data, state_size );


   double ies_max_step=ies_enkf_config_get_ies_max_steplength(ies_config);
   double ies_min_step=ies_enkf_config_get_ies_min_steplength(ies_config);
   double ies_decline_step=ies_enkf_config_get_ies_dec_steplength(ies_config);
   if (ies_decline_step < 1.1) 
      ies_decline_step=1.1;
   double ies_steplength=ies_min_step + (ies_max_step - ies_min_step)*pow(2,-(iteration_nr-1)/(ies_decline_step-1));

   log_fp = ies_enkf_data_open_log(data);

   fprintf(log_fp,"\n\n\n***********************************************************************\n");
   fprintf(log_fp,"IES Iteration   = %d\n", iteration_nr);
   fprintf(log_fp,"----ies_steplength  = %f --- a=%f b=%f c=%f\n", ies_steplength, ies_max_step, ies_min_step, ies_decline_step);
   fprintf(log_fp,"----ies_inversion   = %d\n", ies_inversion);
   fprintf(log_fp,"----ies_debug       = %d\n", ies_debug);
   fprintf(log_fp,"----truncation      = %f %d\n", truncation, subspace_dimension);
   bool dbg = ies_enkf_config_get_ies_debug( ies_config ) ;

/***************************************************************************************************************/
/* Counting number of active observations for current iteration. 
   If the observations have been used in previous iterations they are contained in data->E0.
   If they are introduced in the current iteration they will be augmented to data->E.  */
   ies_enkf_data_store_initialE(data, Ein);
   ies_enkf_data_augment_initialE(data, Ein);

   double nsc = 1.0/sqrt(ens_size - 1.0);

/* dimensions for printing */
   int m_nrobs      = util_int_min(nrobs_inp     -1,7);
   int m_ens_size   = util_int_min(ens_size  -1,16);
   int m_state_size = util_int_min(state_size-1,3);

/***************************************************************************************************************/
   ies_enkf_data_allocateW(data, ens_size_msk);
   ies_enkf_data_store_initialA(data, A);


/***************************************************************************************************************/
   fprintf(log_fp,"----active ens_size  = %d, total ens_size_msk = %d\n", ens_size,ens_size_msk);
   fprintf(log_fp,"----active nrobs_inp = %d, total     nrobs_msk= %d\n", nrobs_inp, nrobs_msk );
   fprintf(log_fp,"----active state_size= %d\n", state_size );



/***************************************************************************************************************/
/* Print initial observation mask */
   printf_mask(log_fp, "obsmask_0:", ies_enkf_data_get_obs_mask0(data));

/***************************************************************************************************************/
/* Print Current observation mask */
   printf_mask(log_fp, "obsmask_i:", ies_enkf_data_get_obs_mask(data));

/***************************************************************************************************************/
/* Print Current ensemble mask */
   printf_mask(log_fp, "ensmask_i:", ies_enkf_data_get_ens_mask(data));

/***************************************************************************************************************
* Re-structure input matrices according to new active obs_mask and ens_size.
*     Allocates the local matrices to be used.
*     Copies the initial measurement perturbations for the active observations into the current E matrix.
*     Copies the inputs in D, Y and R into their local representations
*/
   matrix_type * Y   = matrix_alloc_copy( Yin );
   matrix_type * E   = matrix_alloc( nrobs_inp, ens_size );
   matrix_type * D   = matrix_alloc_copy( Din );
   matrix_type * R   = matrix_alloc_copy( Rin );

/* Subtract new measurement perturbations              D=D-E    */
   matrix_inplace_sub(D,Ein);

/* Extract active observations and realizations for D, E, Y, and R    */
   ies_enkf_linalg_extract_active(data, E, log_fp, dbg);

/* Add old measurement perturbations */
   matrix_inplace_add(D,E);          

   matrix_type * A0  = matrix_alloc( state_size, ens_size );  // Temporary ensemble matrix
   matrix_type * W0  = matrix_alloc( ens_size , ens_size  );  // Coefficient matrix
   matrix_type * W   = matrix_alloc( ens_size , ens_size  );  // Coefficient matrix
   matrix_type * H   = matrix_alloc( nrobs_inp, ens_size  );  // Innovation vector "H= S*W+D-Y"
   matrix_type * S   = matrix_alloc( nrobs_inp, ens_size );   // Predicted ensemble anomalies scaled with inv(Omeaga)
   matrix_type * X   = matrix_alloc( ens_size, ens_size  );   // Final transform matrix

   if (dbg) matrix_pretty_fprint_submat(A,"Ain","%11.5f",log_fp,0,m_state_size,0,m_ens_size) ;
   if (dbg) fprintf(log_fp,"Computed matrices\n");

/***************************************************************************************************************
*  Subtract mean of predictions to generate predicted ensemble anomaly matrix                 (Line 5)
*/
   matrix_subtract_row_mean( Y );   // Y=Y*(I-(1/ens_size)*11)
   matrix_scale(Y,nsc);             // Y=Y / sqrt(ens_size-1)
   if (dbg) matrix_pretty_fprint_submat(Y,"Y","%11.5f",log_fp,0,m_nrobs,0,m_ens_size) ;


/***************************************************************************************************************
*  COMPUTING THE PROJECTION Y= Y * (Ai^+ * Ai) (only used when state_size < ens_size-1)    */
   if (ies_enkf_config_get_ies_aaprojection(ies_config) && (state_size <= (ens_size - 1))) {
      ies_enkf_linalg_compute_AA_projection(A, Y, log_fp, dbg);
   }

/***************************************************************************************************************
*  COPY ACTIVE REALIZATIONS FROM data->W to W0 */
   ies_enkf_linalg_extract_active_W(data, W0, log_fp, dbg);

/***************************************************************************************************************
*  When solving the system S = Y inv(Omega) we write                                           (line 6)
*     Omega^T S^T = Y^T
*/
   ies_linalg_solve_S(W0, Y, S, rcond, log_fp, dbg);

/***************************************************************************************************************
*  INNOVATION H = S*W + D - Y   from Eq. (47)                                                  (Line 8)    */
   matrix_assign(H,D) ;                            // H=D=dobs + E - Y
   matrix_dgemm(H,S,W0,false,false,1.0,1.0);       // H=S*W + H

   if (dbg) matrix_pretty_fprint_submat(H,"H","%11.5f",log_fp,0,m_nrobs,0,m_ens_size) ;

/* Store previous W for convergence test */
   matrix_assign(W,W0);

/***************************************************************************************************************
* COMPUTE NEW UPDATED W                                                                        (Line 9)
*     W = W + ies_steplength * ( W - S'*(S*S'+R)^{-1} H )          (a)
*  which in the case when R=I can be rewritten as
*     W = W + ies_steplength * ( W - (S'*S + I)^{-1} * S' * H )    (b)
*
*  With R=I the subspace inversion (ies_inversion=1) solving Eq. (a) with singular value
*  trucation=1.000 gives exactly the same solution as the exact inversion (ies_inversion=0).
*
*  Using ies_inversion=IES_INVERSION_SUBSPACE_EXACT_R(2), and a step length of 1.0,
*  one update gives identical result to ENKF_STD as long as the same SVD
*  truncation is used.
*
*  With very large data sets it is likely that the inversion becomes poorly
*  conditioned and a trucation=1.000 is not a good choice. In this case the
*  ies_inversion > 0 and EnKF_truncation set to 0.99 or so, should stabelize
*  the algorithm.
*
*  Using ies_inversion=IES_INVERSION_SUBSPACE_EE_R(3) and
*  ies_inversion=IES_INVERSION_SUBSPACE_RE(2) gives identical results but
*  ies_inversion=IES_INVERSION_SUBSPACE_RE is much faster (N^2m) than
*  ies_inversion=IES_INVERSION_SUBSPACE_EE_R (Nm^2).

   See the enum: ies_inverson in ies_enkf_config.h:

   ies_inversion=IES_INVERSION_EXACT(0)            -> exact inversion from (b) with exact R=I
   ies_inversion=IES_INVERSION_SUBSPACE_EXACT_R(1) -> subspace inversion from (a) with exact R
   ies_inversion=IES_INVERSION_SUBSPACE_EE_R(2)    -> subspace inversion from (a) with R=EE
   ies_inversion=IES_INVERSION_SUBSPACE_RE(3)      -> subspace inversion from (a) with R represented by E
*/
   if (ies_inversion != IES_INVERSION_EXACT){
      fprintf(log_fp,"Subspace inversion. (ies_inversion=%d)\n",ies_inversion);
      ies_enkf_linalg_subspace_inversion(W0, ies_inversion, E, R, S, H, truncation, ies_steplength, subspace_dimension, log_fp, dbg);
   } else if (ies_inversion == IES_INVERSION_EXACT) {
      fprintf(log_fp,"Exact inversion using diagonal R=I. (ies_inversion=%d)\n",ies_inversion);
      ies_enkf_linalg_exact_inversion( W0, ies_inversion, S, H, ies_steplength, log_fp, dbg);
   }

   if (dbg) matrix_pretty_fprint_submat(W0,"Updated W","%11.5f",log_fp,0,m_ens_size,0,m_ens_size) ;

/* Store active realizations from W0 to data->W */
   ies_enkf_linalg_store_active_W(data, W0, log_fp, dbg);

/***************************************************************************************************************
*  CONSTRUCT TRANFORM MATRIX X FOR CURRENT ITERATION                                         (Line 10)
*     X= I + W/sqrt(N-1)          */
   matrix_assign(X,W0);
   matrix_scale(X,nsc);
   for (int i = 0; i < ens_size; i++){
      matrix_iadd(X,i,i,1.0);
   }
   if (dbg) matrix_pretty_fprint_submat(X,"X","%11.5f",log_fp,0,m_ens_size-1,0,m_ens_size);

/***************************************************************************************************************
*  COMPUTE NEW ENSEMBLE SOLUTION FOR CURRENT ITERATION  Ei=A0*X                              (Line 11)   */
   matrix_pretty_fprint_submat(A,"A^f","%11.5f",log_fp,0,m_state_size,0,m_ens_size);
   ies_enkf_linalg_extract_active_A0(data, A0, log_fp, dbg);
   matrix_matmul(A,A0,X);
   matrix_pretty_fprint_submat(A,"A^a","%11.5f",log_fp,0,m_state_size,0,m_ens_size);

/***************************************************************************************************************
*  COMPUTE ||W0 - W|| AND EVALUATE COST FUNCTION FOR PREVIOUS ITERATE                        (Line 12)   */
   matrix_type * DW  = matrix_alloc( ens_size , ens_size  ); 
   matrix_sub(DW,W0,W);
   teclog(W,D,DW,"iesteclog.dat",ens_size,iteration_nr, rcond, nrsing, nrobs_inp);
   teccost(W,D,"costf.dat",ens_size,iteration_nr);

/* DONE *********************************************************************************************************/

   ies_enkf_data_fclose_log(data);

   matrix_free( Y  );
   matrix_free( D  );
   matrix_free( E  );
   matrix_free( R  );
   matrix_free( A0 );
   matrix_free( W0 );
   matrix_free( W );
   matrix_free( DW );
   matrix_free( H  );
   matrix_free( S  );
   matrix_free( X  );
}


/***************************************************************************************************************/
/* ies_enkf linalg functions */
/***************************************************************************************************************/
/* Extract active observations and realizations for D, E, Y, and R.
*  IES works from an initial set of active realizations and measurements.
*  In the first iteration we store the measurement perturbations from the initially active observations.
*  In the subsequent iterations we include the new observations that may be introduced by ERT including their new
*  measurement perturbations, and we reuse the intially stored measurement perturbations.
*  Additionally we may loose realizations during the iteratitions, and we extract measurement perturbations for
*  only the active realizations.
*/
void ies_enkf_linalg_extract_active(const ies_enkf_data_type * data,
                                    matrix_type * E,
                                    FILE * log_fp,
                                    bool dbg){

   const bool_vector_type * obs_mask  = ies_enkf_data_get_obs_mask(data);
   const bool_vector_type * ens_mask  = ies_enkf_data_get_ens_mask(data);

   int obs_size_msk  = ies_enkf_data_get_obs_mask_size(data); // Total number of observations
   int ens_size_msk  = ies_enkf_data_get_ens_mask_size(data);   // Total number of realizations

   const matrix_type * dataE = ies_enkf_data_getE(data);

   int m=-1;  // counter for currently active measurements
   for (int iobs = 0; iobs < obs_size_msk; iobs++){
      if ( bool_vector_iget(obs_mask,iobs) ) {
         m++;
         fprintf(log_fp,"ies_enkf_linalg_extract_active iobs=%d m=%d)\n",iobs,m);
         int i=-1;
         for (int iens = 0; iens < ens_size_msk; iens++){
            if ( bool_vector_iget(ens_mask,iens) ){
               i++;
               matrix_iset_safe(E,m,i,matrix_iget(dataE,iobs,iens)) ;
            }
         }
      }
   }

   if (dbg) {
      int m_nrobs      = util_int_min(matrix_get_rows( E )   -1,7);
      int m_ens_size   = util_int_min(matrix_get_columns( E )-1,16);
      matrix_pretty_fprint_submat(E,"E","%11.5f",log_fp,0,m_nrobs,0,m_ens_size) ;
   }

}



/*  COMPUTING THE PROJECTION Y= Y * (Ai^+ * Ai) (only used when state_size < ens_size-1)    */
void ies_enkf_linalg_compute_AA_projection(const matrix_type * A,
                                                 matrix_type * Y,
                                                 FILE * log_fp,
                                                 bool dbg){
   int ens_size      = matrix_get_columns( A );            
   int state_size    = matrix_get_rows( A );
   int nrobs         = matrix_get_rows( Y );

   int m_nrobs      = util_int_min(nrobs     -1,7);
   int m_ens_size   = util_int_min(ens_size  -1,16);
   int m_state_size = util_int_min(state_size-1,3);

   fprintf(log_fp,"Activating AAi projection for Y\n");
   double      * eig = (double*)util_calloc( ens_size , sizeof * eig);
   matrix_type * Ai    = matrix_alloc_copy( A );
   matrix_type * AAi   = matrix_alloc( ens_size, ens_size  );
   matrix_subtract_row_mean(Ai);
   matrix_type * VT    = matrix_alloc( state_size, ens_size  );
   matrix_dgesvd(DGESVD_NONE , DGESVD_MIN_RETURN , Ai , eig , NULL , VT);
   if (dbg) matrix_pretty_fprint_submat(VT,"VT","%11.5f",log_fp,0,m_state_size-1,0,m_ens_size) ;
   matrix_dgemm(AAi,VT,VT,true,false,1.0,0.0);
   if (dbg) matrix_pretty_fprint_submat(AAi,"AAi","%11.5f",log_fp,0,m_ens_size-1,0,m_ens_size);
   matrix_inplace_matmul(Y,AAi);
   matrix_free(Ai);
   matrix_free(AAi);
   matrix_free(VT);
   free(eig);
   if (dbg) matrix_pretty_fprint_submat(Y,"Yprojected","%11.5f",log_fp,0,m_nrobs,0,m_ens_size) ;
}



/*
* W is stored for each iteration in data->W. If we loose realizations we copy only the active rows and cols from
* data->W to W0 which is then used in the algorithm.
*/
void ies_enkf_linalg_extract_active_W(const ies_enkf_data_type * data,
                                            matrix_type * W0,
                                            FILE * log_fp,
                                            bool dbg){

   const bool_vector_type * ens_mask = ies_enkf_data_get_ens_mask(data);
   const matrix_type * dataW = ies_enkf_data_getW(data);
   int ens_size_msk  = ies_enkf_data_get_ens_mask_size(data);
   int ens_size      = matrix_get_columns( W0 );            
   int m_ens_size    = util_int_min(ens_size  -1,16);
   int i=-1;
   int j;
   for (int iens=0; iens < ens_size_msk; iens++){
      if ( bool_vector_iget(ens_mask,iens) ){
         i=i+1;
         j=-1;
         for (int jens=0; jens < ens_size_msk; jens++){
            if ( bool_vector_iget(ens_mask,jens) ){
               j=j+1;
               matrix_iset_safe(W0,i,j,matrix_iget(dataW,iens,jens)) ;
            }
         }
      }
   }

   if (ens_size_msk == ens_size){
     fprintf(log_fp,"data->W copied exactly to W0: %d\n",matrix_equal(dataW,W0)) ;
   }

   if (dbg) matrix_pretty_fprint_submat(dataW,"data->W","%11.5f",log_fp,0,m_ens_size-1,0,m_ens_size);
   if (dbg) matrix_pretty_fprint_submat(W0,"W0","%11.5f",log_fp,0,m_ens_size-1,0,m_ens_size);
}

/*
* COMPUTE  Omega= I + W (I-11'/sqrt(ens_size))    from Eq. (36).                                   (Line 6)
*  When solving the system S = Y inv(Omega) we write
*     Omega^T S^T = Y^T
*/
void ies_linalg_solve_S(const matrix_type * W0,
                        const matrix_type * Y,
                              matrix_type * S,
                              double rcond,
                              FILE * log_fp,
                              bool dbg){
   int ens_size      = matrix_get_columns( W0 );            
   int nrobs         = matrix_get_rows( S );            
   int m_ens_size    = util_int_min(ens_size  -1,16);
   int m_nrobs      = util_int_min(nrobs     -1,7);
   double nsc = 1.0/sqrt(ens_size - 1.0);

   matrix_type * YT  = matrix_alloc( ens_size, nrobs     );   // Y^T used in linear solver
   matrix_type * ST  = matrix_alloc( ens_size, nrobs     );   // current S^T used in linear solver
   matrix_type * Omega   = matrix_alloc( ens_size, ens_size  );   // current S^T used in linear solver

/*  Here we compute the W (I-11'/N) / sqrt(N-1)  and transpose it).*/
   matrix_assign(Omega,W0) ;            // Omega=data->W (from previous iteration used to solve for S)
   matrix_subtract_row_mean(Omega);     // Omega=Omega*(I-(1/N)*11')
   matrix_scale(Omega,nsc);             // Omega/sqrt(N-1)
   matrix_inplace_transpose(Omega);     // Omega=transpose(Omega)
   for (int i = 0; i < ens_size; i++){  // Omega=Omega+I
      matrix_iadd(Omega,i,i,1.0);
   }
   if (dbg) matrix_pretty_fprint_submat(Omega,"OmegaT","%11.5f",log_fp,0,m_ens_size,0,m_ens_size) ;
   matrix_transpose(Y,YT);         // RHS stored in YT

/* Solve system                                                                                (Line 7)   */
   fprintf(log_fp,"Solving Omega' S' = Y' using LU factorization:\n");
   matrix_dgesvx(Omega,YT,&rcond);
   fprintf(log_fp,"dgesvx condition number= %12.5e\n",rcond);

   matrix_transpose(YT,S);          // Copy solution to S

   if (dbg) matrix_pretty_fprint_submat(S,"S","%11.5f",log_fp,0,m_nrobs,0,m_ens_size) ;
   matrix_free(Omega);
   matrix_free(ST);
   matrix_free(YT);
}


/*
*  The standard inversion works on the equation
*          S'*(S*S'+R)^{-1} H           (a)
*/
void ies_enkf_linalg_subspace_inversion(matrix_type * W0,
                                        const int ies_inversion,
                                        matrix_type * E,
                                        matrix_type * R,
                                        const matrix_type * S,
                                        const matrix_type * H,
                                        double truncation,
                                        double ies_steplength,
                                        int subspace_dimension,
                                        FILE * log_fp,
                                        bool dbg){

   int ens_size      = matrix_get_columns( S );            
   int m_ens_size    = util_int_min(ens_size  -1,16);
   int nrobs         = matrix_get_rows( S );            
   int m_nrobs       = util_int_min(nrobs     -1,7);
   int nrmin  = util_int_min( ens_size , nrobs);
   double nsc = 1.0/sqrt(ens_size - 1.0);
   matrix_type * X1  = matrix_alloc( nrobs   , nrmin     );   // Used in subspace inversion
   matrix_type * X3  = matrix_alloc( nrobs   , ens_size  );   // Used in subspace inversion
   double      * eig = (double*)util_calloc( ens_size , sizeof * eig);

   fprintf(log_fp,"Subspace inversion. (ies_inversion=%d)\n",ies_inversion);

   if (ies_inversion == IES_INVERSION_SUBSPACE_RE){
      fprintf(log_fp,"Subspace inversion using E to represent errors. (ies_inversion=%d)\n",ies_inversion);
      matrix_scale(E,nsc);
      enkf_linalg_lowrankE( S , E , X1 , eig , truncation , subspace_dimension);
   } else if (ies_inversion == IES_INVERSION_SUBSPACE_EE_R){
      fprintf(log_fp,"Subspace inversion using ensemble generated full R=EE. (ies_inversion=%d)'\n",ies_inversion);
      matrix_scale(E,nsc);
      matrix_type * Et = matrix_alloc_transpose( E );
      matrix_type * Cee = matrix_alloc_matmul( E , Et );
      matrix_scale(Cee,nsc*nsc); // since enkf_linalg_lowrankCinv solves (SS' + (N-1) R)^{-1}
      if (dbg) matrix_pretty_fprint_submat(Cee,"Cee","%11.5f",log_fp,0,m_nrobs,0,m_nrobs) ;
      enkf_linalg_lowrankCinv( S , Cee , X1 , eig , truncation , subspace_dimension);
      matrix_free( Et );
      matrix_free( Cee );
   } else if (ies_inversion == IES_INVERSION_SUBSPACE_EXACT_R){
      fprintf(log_fp,"Subspace inversion using 'exact' full R. (ies_inversion=%d)\n",ies_inversion);
      matrix_scale(R,nsc*nsc); // since enkf_linalg_lowrankCinv solves (SS' + (N-1) R)^{-1}
      if (dbg) matrix_pretty_fprint_submat(R,"R","%11.5f",log_fp,0,m_nrobs,0,m_nrobs) ;
      enkf_linalg_lowrankCinv( S , R , X1 , eig , truncation , subspace_dimension);
   }

   int nrsing=0;
   fprintf(log_fp,"\nEig:\n");
   for (int i=0;i<nrmin;i++){
      fprintf(log_fp," %f ", eig[i]);
      if ((i+1)%20 == 0) fprintf(log_fp,"\n") ;
      if (eig[i] < 1.0) nrsing+=1;
   }
   fprintf(log_fp,"\n");

/*    X3 = X1 * diag(eig) * X1' * H (Similar to Eq. 14.31, Evensen (2007))                                  */
   enkf_linalg_genX3(X3 , X1 , H , eig);

   if (dbg) matrix_pretty_fprint_submat(X1,"X1","%11.5f",log_fp,0,m_nrobs,0,util_int_min(m_nrobs,nrmin-1)) ;
   if (dbg) matrix_pretty_fprint_submat(X3,"X3","%11.5f",log_fp,0,m_nrobs,0,m_ens_size) ;

/*    Update data->W = (1-ies_steplength) * data->W +  ies_steplength * S' * X3                          (Line 9)    */
   matrix_dgemm(W0 , S , X3 , true , false , ies_steplength , 1.0-ies_steplength);

   matrix_free( X1 );
   matrix_free( X3 );
   free(eig);
}

/*
*  The standard inversion works on the equation
*          S'*(S*S'+R)^{-1} H           (a)
*  which in the case when R=I can be rewritten as
*          (S'*S + I)^{-1} * S' * H     (b)
*/
void ies_enkf_linalg_exact_inversion(matrix_type * W0,
                                     const int ies_inversion,
                                     const matrix_type * S,
                                     const matrix_type * H,
                                     double ies_steplength,
                                     FILE * log_fp,
                                     bool dbg){
   int ens_size      = matrix_get_columns( S );            

   fprintf(log_fp,"Exact inversion using diagonal R=I. (ies_inversion=%d)\n",ies_inversion);
   matrix_type * Z      = matrix_alloc( ens_size , ens_size  );  // Eigen vectors of S'S+I
   matrix_type * ZtStH  = matrix_alloc( ens_size , ens_size );
   matrix_type * StH    = matrix_alloc( ens_size , ens_size );
   matrix_type * StS    = matrix_alloc( ens_size , ens_size );
   double      * eig = (double*)util_calloc( ens_size , sizeof * eig);

   matrix_diag_set_scalar(StS,1.0);
   matrix_dgemm(StS,S,S,true,false,1.0,1.0);
   matrix_dgesvd(DGESVD_ALL , DGESVD_NONE , StS , eig , Z , NULL);

   matrix_dgemm(StH,S,H,true,false,1.0,0.0);
   matrix_dgemm(ZtStH,Z,StH,true,false,1.0,0.0);

   for (int i=0;i<ens_size;i++){
      eig[i]=1.0/eig[i] ;
      matrix_scale_row(ZtStH,i,eig[i]);
   }

   fprintf(log_fp,"\nEig:\n");
   for (int i=0;i<ens_size;i++){
      fprintf(log_fp," %f ", eig[i]);
      if ((i+1)%20 == 0) fprintf(log_fp,"\n") ;
   }
   fprintf(log_fp,"\n");
/*    Update data->W = (1-ies_steplength) * data->W +  ies_steplength * Z * (Lamda^{-1}) Z' S' H         (Line 9)    */
      matrix_dgemm(W0 , Z , ZtStH , false , false , ies_steplength , 1.0-ies_steplength);

      matrix_free(Z);
      matrix_free(ZtStH);
      matrix_free(StH);
      matrix_free(StS);
      free(eig);
}


/*
* the updated W is stored for each iteration in data->W. If we have lost realizations we copy only the active rows and cols from
* W0 to data->W which is then used in the algorithm.  (note the definition of the pointer dataW to data->W)
*/
void ies_enkf_linalg_store_active_W(ies_enkf_data_type * data,
                                    const matrix_type * W0,
                                    FILE * log_fp,
                                    bool dbg){
   int ens_size_msk  = ies_enkf_data_get_ens_mask_size(data);
   int ens_size      = matrix_get_columns( W0 );            
   int i=0;
   int j;
   matrix_type * dataW = ies_enkf_data_getW(data);
   const bool_vector_type * ens_mask = ies_enkf_data_get_ens_mask(data);
   matrix_set(dataW , 0.0) ;
   for (int iens=0; iens < ens_size_msk; iens++){
      if ( bool_vector_iget(ens_mask,iens) ){
         j=0;
         for (int jens=0; jens < ens_size_msk; jens++){
            if ( bool_vector_iget(ens_mask,jens) ){
              matrix_iset_safe(dataW,iens,jens,matrix_iget(W0,i,j));
              j += 1;
            }
         }
         i += 1;
      }
   }

   if (ens_size_msk == ens_size){
     fprintf(log_fp,"W0 copied exactly to data->W: %d\n",matrix_equal(dataW,W0)) ;
   }
}

/*
* Extract active realizations from the initially stored ensemble.
*/
void ies_enkf_linalg_extract_active_A0(const ies_enkf_data_type * data,
                                       matrix_type * A0,
                                       FILE * log_fp,
                                       bool dbg){
   int ens_size_msk  = ies_enkf_data_get_ens_mask_size(data);
   int ens_size      = matrix_get_columns( A0 );            
   int state_size    = matrix_get_rows( A0 );
   int m_ens_size    = util_int_min(ens_size  -1,16);
   int m_state_size = util_int_min(state_size-1,3);
   int i=-1;
   const bool_vector_type * ens_mask = ies_enkf_data_get_ens_mask(data);
   const matrix_type * dataA0 = ies_enkf_data_getA0(data);
   matrix_pretty_fprint_submat(dataA0,"data->A0","%11.5f",log_fp,0,m_state_size,0,m_ens_size);
   for (int iens=0; iens < ens_size_msk; iens++){
      if ( bool_vector_iget(ens_mask,iens) ){
         i=i+1;
         matrix_copy_column(A0,dataA0,i,iens);
      }
   }

   if (ens_size_msk == ens_size){
     fprintf(log_fp,"data->A0 copied exactly to A0: %d\n",matrix_equal(dataA0,A0)) ;
   }
}



//**********************************************
// Set / Get basic types
//**********************************************
bool ies_enkf_set_int( void * arg , const char * var_name , int value) {
  ies_enkf_data_type * module_data = ies_enkf_data_safe_cast( arg );
  ies_enkf_config_type * config = ies_enkf_data_get_config( module_data );
  {
    bool name_recognized = true;

    if (strcmp( var_name , ENKF_SUBSPACE_DIMENSION_KEY) == 0)
      ies_enkf_config_set_enkf_subspace_dimension(config , value);
    else if (strcmp( var_name , ITER_KEY) == 0)
      ies_enkf_data_set_iteration_nr( module_data , value );
    else if (strcmp( var_name , IES_INVERSION_KEY) == 0)  // This should probably translate string value - now it goes directly on the value of the ies_inversion_type enum.
      ies_enkf_config_set_ies_inversion( config , value );
    else
      name_recognized = false;

    return name_recognized;
  }
}

int ies_enkf_get_int( const void * arg, const char * var_name) {
  const ies_enkf_data_type * module_data = ies_enkf_data_safe_cast_const( arg );
  const ies_enkf_config_type * ies_config = ies_enkf_data_get_config( module_data );
  {
    if (strcmp(var_name , ITER_KEY) == 0)
      return ies_enkf_data_get_iteration_nr(module_data);
    else if (strcmp(var_name , ENKF_SUBSPACE_DIMENSION_KEY) == 0)
      return ies_enkf_config_get_enkf_subspace_dimension(ies_config);
    else if (strcmp(var_name , IES_INVERSION_KEY) == 0)
      return ies_enkf_config_get_ies_inversion(ies_config);
    else
      return -1;
  }
}

bool ies_enkf_set_string( void * arg , const char * var_name , const char * value) {
  ies_enkf_data_type * module_data = ies_enkf_data_safe_cast( arg );
  ies_enkf_config_type * ies_config = ies_enkf_data_get_config( module_data );
  {
    bool name_recognized = true;

    if (strcmp( var_name , IES_LOGFILE_KEY) == 0)
      ies_enkf_config_set_ies_logfile( ies_config , value );
    else
      name_recognized = false;

    return name_recognized;
  }
}


bool ies_enkf_set_bool( void * arg , const char * var_name , bool value) {
  ies_enkf_data_type * module_data = ies_enkf_data_safe_cast( arg );
  ies_enkf_config_type * ies_config = ies_enkf_data_get_config( module_data );
  {
    bool name_recognized = true;

    if (strcmp( var_name , IES_SUBSPACE_KEY) == 0)
      ies_enkf_config_set_ies_subspace( ies_config , value);
    else if (strcmp( var_name , IES_DEBUG_KEY) == 0)
      ies_enkf_config_set_ies_debug( ies_config , value );
    else if (strcmp( var_name , IES_AAPROJECTION_KEY) == 0)
      ies_enkf_config_set_ies_aaprojection( ies_config , value );
    else
      name_recognized = false;

    return name_recognized;
  }
}

bool ies_enkf_get_bool( const void * arg, const char * var_name) {
  const ies_enkf_data_type * module_data = ies_enkf_data_safe_cast_const( arg );
  const ies_enkf_config_type * ies_config = ies_enkf_data_get_config( module_data );
  {
    if (strcmp(var_name , IES_SUBSPACE_KEY) == 0)
      return ies_enkf_config_get_ies_subspace( ies_config );
    else if (strcmp(var_name , IES_DEBUG_KEY) == 0)
      return ies_enkf_config_get_ies_debug( ies_config );
    else if (strcmp(var_name , IES_AAPROJECTION_KEY) == 0)
      return ies_enkf_config_get_ies_aaprojection( ies_config );
    else
       return false;
  }
}




bool ies_enkf_set_double( void * arg , const char * var_name , double value) {
  ies_enkf_data_type * module_data = ies_enkf_data_safe_cast( arg );
  ies_enkf_config_type * ies_config = ies_enkf_data_get_config( module_data );
  {
    bool name_recognized = true;

    if (strcmp( var_name , ENKF_TRUNCATION_KEY) == 0)
      ies_enkf_config_set_truncation( ies_config , value );
    else if (strcmp( var_name , IES_MAX_STEPLENGTH_KEY) == 0)
      ies_enkf_config_set_ies_max_steplength( ies_config , value );
    else if (strcmp( var_name , IES_MIN_STEPLENGTH_KEY) == 0)
      ies_enkf_config_set_ies_min_steplength( ies_config , value );
    else if (strcmp( var_name , IES_DEC_STEPLENGTH_KEY) == 0)
      ies_enkf_config_set_ies_dec_steplength( ies_config , value );
    else
      name_recognized = false;

    return name_recognized;
  }
}

double ies_enkf_get_double( const void * arg, const char * var_name) {
  const ies_enkf_data_type * module_data = ies_enkf_data_safe_cast_const( arg );
  const ies_enkf_config_type * ies_config = ies_enkf_data_get_config( module_data );
  {
    if (strcmp(var_name , ENKF_TRUNCATION_KEY) == 0)
      return ies_enkf_config_get_truncation( ies_config );
    if (strcmp(var_name , IES_MAX_STEPLENGTH_KEY) == 0)
      return ies_enkf_config_get_ies_max_steplength(ies_config);
    if (strcmp(var_name , IES_MIN_STEPLENGTH_KEY) == 0)
      return ies_enkf_config_get_ies_min_steplength(ies_config);
    if (strcmp(var_name , IES_DEC_STEPLENGTH_KEY) == 0)
      return ies_enkf_config_get_ies_dec_steplength(ies_config);
    return -1;
  }
}


long ies_enkf_get_options( void * arg , long flag ) {
  ies_enkf_data_type * module_data = ies_enkf_data_safe_cast( arg );
  const ies_enkf_config_type * ies_config = ies_enkf_data_get_config( module_data );
  {
    return ies_enkf_config_get_option_flags( ies_config );
  }
}

bool ies_enkf_has_var( const void * arg, const char * var_name) {
  {
    if (strcmp(var_name , ITER_KEY) == 0)
      return true;
    else if (strcmp(var_name , IES_MAX_STEPLENGTH_KEY) == 0)
      return true;
    else if (strcmp(var_name , IES_MIN_STEPLENGTH_KEY) == 0)
      return true;
    else if (strcmp(var_name , IES_DEC_STEPLENGTH_KEY) == 0)
      return true;
    else if (strcmp(var_name , IES_SUBSPACE_KEY) == 0)
      return true;
    else if (strcmp(var_name , IES_INVERSION_KEY) == 0)
      return true;
    else if (strcmp(var_name , IES_LOGFILE_KEY) == 0)
      return true;
    else if (strcmp(var_name , IES_DEBUG_KEY) == 0)
      return true;
    else if (strcmp(var_name , IES_AAPROJECTION_KEY) == 0)
      return true;
    else if (strcmp(var_name , ENKF_TRUNCATION_KEY) == 0)
      return true;
    else if (strcmp(var_name , ENKF_SUBSPACE_DIMENSION_KEY) == 0)
      return true;
    else
      return false;
  }
}

void * ies_enkf_get_ptr( const void * arg , const char * var_name ) {
  const ies_enkf_data_type * module_data = ies_enkf_data_safe_cast_const( arg );
  const ies_enkf_config_type * ies_config = ies_enkf_data_get_config( module_data );
  {
    if (strcmp(var_name , IES_LOGFILE_KEY) == 0)
      return (void *) ies_enkf_config_get_ies_logfile( ies_config );
    else
      return NULL;
  }
}


//**********************************************
// Symbol table
//**********************************************
#ifdef INTERNAL_LINK
#define LINK_NAME IES_ENKF
#else
#define LINK_NAME EXTERNAL_MODULE_SYMBOL
#endif


analysis_table_type LINK_NAME = {
  .name            = "IES_ENKF",
  .initX           = NULL,
  .updateA         = ies_enkf_updateA,
  .init_update     = ies_enkf_init_update,
  .complete_update = NULL,
  .alloc           = ies_enkf_data_alloc,
  .freef           = ies_enkf_data_free,
  .has_var         = ies_enkf_has_var,
  .set_int         = ies_enkf_set_int ,
  .set_double      = ies_enkf_set_double ,
  .set_bool        = ies_enkf_set_bool ,
  .set_string      = ies_enkf_set_string ,
  .get_options     = ies_enkf_get_options ,
  .get_int         = ies_enkf_get_int,
  .get_double      = ies_enkf_get_double,
  .get_bool        = ies_enkf_get_bool ,
  .get_ptr         = ies_enkf_get_ptr ,
};

