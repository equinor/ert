/*
   Copyright (C) 2011  Statoil ASA, Norway. 
    
   The file 'rml_enkf_imodel.c' is part of ERT - Ensemble based Reservoir Tool. 
    
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

#include <ert/util/util.h>
#include <ert/util/type_macros.h>
#include <ert/util/rng.h>
#include <ert/util/matrix.h>
#include <ert/util/matrix_blas.h>
#include <ert/util/bool_vector.h>

#include <ert/analysis/analysis_module.h>
#include <ert/analysis/analysis_table.h>
#include <ert/analysis/enkf_linalg.h>
#include <ert/analysis/std_enkf.h>

#include <rml_enkf_common.h>

/*
  A random 'magic' integer id which is used for run-time type checking
  of the input data. 
*/
#define RML_ENKF_IMODEL_TYPE_ID 261123

typedef struct rml_enkf_imodel_data_struct rml_enkf_imodel_data_type;


/*
  Observe that only one of the settings subspace_dimension and
  truncation can be valid at a time; otherwise the svd routine will
  fail. This implies that the set_truncation() and
  set_subspace_dimension() routines will set one variable, AND
  INVALIDATE THE OTHER. For most situations this will be OK, but if
  you have repeated calls to both of these functions the end result
  might be a surprise.  
*/
#define INVALID_SUBSPACE_DIMENSION  -1
#define INVALID_TRUNCATION          -1
#define DEFAULT_SUBSPACE_DIMENSION  INVALID_SUBSPACE_DIMENSION
#define DEFAULT_USE_PRIOR           true
#define DEFAULT_LAMBDA_INCREASE_FACTOR 4
#define DEFAULT_LAMBDA_REDUCE_FACTOR   0.1
#define DEFAULT_LAMBDA0                -1


#define  USE_PRIOR_KEY               "USE_PRIOR"
#define  LAMBDA_REDUCE_FACTOR_KEY    "LAMBDA_REDUCE"
#define  LAMBDA_INCREASE_FACTOR_KEY  "LAMBDA_INCREASE"
#define  LAMBDA0_KEY                 "LAMBDA0"
#define  ITER_KEY                    "ITER"




/*
  The configuration data used by the rml_enkf_imodel module is contained in a
  rml_enkf_imodel_data_struct instance. The data type used for the rml_enkf_imodel
  module is quite simple; with only a few scalar variables, but there
  are essentially no limits to what you can pack into such a datatype.

  All the functions in the module have a void pointer as the first
  argument, this will immediately be casted to a rml_enkf_imodel_data_type
  instance, to get some type safety the UTIL_TYPE_ID system should be
  used (see documentation in util.h)

  The data structure holding the data for your analysis module should
  be created and initialized by a constructor, which should be
  registered with the '.alloc' element of the analysis table; in the
  same manner the desctruction of this data should be handled by a
  destructor or free() function registered with the .freef field of
  the analysis table.
*/




struct rml_enkf_imodel_data_struct {
  UTIL_TYPE_ID_DECLARATION;
  double    truncation;            // Controlled by config key: ENKF_TRUNCATION_KEY
  int       subspace_dimension;    // Controlled by config key: ENKF_NCOMP_KEY (-1: use Truncation instead)
  long      option_flags;
  int       iteration_nr;          // Keep track of the outer iteration loop
  double    lambda;                 // parameter to control the search direction in Marquardt levenberg optimization 
  double    lambda0;
  double    Sk;                    // Objective function value
  double    Std;                   // Standard Deviation of the Objective function
  double  * Csc;
  matrix_type *Am;
  matrix_type *active_prior;
  matrix_type *prior0;
  matrix_type *state;
  bool_vector_type * ens_mask;
  bool use_prior;
  double lambda_reduce_factor;
  double lambda_increase_factor;
};


/*
  This is a macro which will expand to generate a function:

     rml_enkf_imodel_data_type * rml_enkf_imodel_data_safe_cast( void * arg ) {}

  which is used for runtime type checking of all the functions which
  accept a void pointer as first argument. 
*/
static UTIL_SAFE_CAST_FUNCTION( rml_enkf_imodel_data , RML_ENKF_IMODEL_TYPE_ID )
static UTIL_SAFE_CAST_FUNCTION_CONST( rml_enkf_imodel_data , RML_ENKF_IMODEL_TYPE_ID )


double rml_enkf_imodel_get_truncation( rml_enkf_imodel_data_type * data ) {
  return data->truncation;
}

int rml_enkf_imodel_get_subspace_dimension( rml_enkf_imodel_data_type * data ) {
  return data->subspace_dimension;
}

void rml_enkf_imodel_set_truncation( rml_enkf_imodel_data_type * data , double truncation ) {
  data->truncation = truncation;
  if (truncation > 0.0)
    data->subspace_dimension = INVALID_SUBSPACE_DIMENSION;
}

void rml_enkf_imodel_set_lambda0( rml_enkf_imodel_data_type * data , double increase_factor) {
  data->lambda0 = increase_factor;
}


void rml_enkf_imodel_set_lambda_increase_factor( rml_enkf_imodel_data_type * data , double increase_factor) {
  data->lambda_increase_factor = increase_factor;
}

void rml_enkf_imodel_set_lambda_reduce_factor( rml_enkf_imodel_data_type * data , double reduce_factor) {
  data->lambda_reduce_factor = reduce_factor;
}

double rml_enkf_imodel_get_lambda_increase_factor( const rml_enkf_imodel_data_type * data ) {
  return data->lambda_increase_factor;
}

double rml_enkf_imodel_get_lambda_reduce_factor( const rml_enkf_imodel_data_type * data ) {
  return data->lambda_reduce_factor;
}

double rml_enkf_imodel_get_lambda0( const rml_enkf_imodel_data_type * data ) {
  return data->lambda0;
}


bool rml_enkf_imodel_get_use_prior( const rml_enkf_imodel_data_type * data ) {
  return data->use_prior;
}


void rml_enkf_imodel_set_use_prior( rml_enkf_imodel_data_type * data , bool use_prior) {
  data->use_prior = use_prior;
}

void rml_enkf_imodel_set_subspace_dimension( rml_enkf_imodel_data_type * data , int subspace_dimension) {
  data->subspace_dimension = subspace_dimension;
  if (subspace_dimension > 0)
    data->truncation = INVALID_TRUNCATION;
}


void rml_enkf_imodel_set_iteration_nr( rml_enkf_imodel_data_type * data , int iteration_nr) {
  data->iteration_nr = iteration_nr;
}


int rml_enkf_imodel_get_iteration_nr( const rml_enkf_imodel_data_type * data ) {
  return data->iteration_nr;
}





void * rml_enkf_imodel_data_alloc( rng_type * rng) {
  rml_enkf_imodel_data_type * data = util_malloc( sizeof * data);
  UTIL_TYPE_ID_INIT( data , RML_ENKF_IMODEL_TYPE_ID );
  
  rml_enkf_imodel_set_truncation( data , DEFAULT_ENKF_TRUNCATION_ );
  rml_enkf_imodel_set_subspace_dimension( data , DEFAULT_SUBSPACE_DIMENSION );
  rml_enkf_imodel_set_use_prior( data , DEFAULT_USE_PRIOR );
  rml_enkf_imodel_set_lambda0( data , DEFAULT_LAMBDA0 );
  data->option_flags = ANALYSIS_NEED_ED + ANALYSIS_UPDATE_A + ANALYSIS_ITERABLE + ANALYSIS_SCALE_DATA;
  data->iteration_nr = 0;
  data->Std          = 0; 
  data->ens_mask     = bool_vector_alloc(0,false);
  data->state        = matrix_alloc(1,1);
  data->active_prior = matrix_alloc(1,1);
  data->prior0       = matrix_alloc(1,1);
  return data;
}


void rml_enkf_imodel_data_free( void * arg ) { 
  rml_enkf_imodel_data_type * data = rml_enkf_imodel_data_safe_cast( arg );

  matrix_free( data->state );
  matrix_free( data->prior0 );
  matrix_free( data->active_prior );

  bool_vector_free( data->ens_mask );
  free( data );
}


static void rml_enkf_imodel_init1__( rml_enkf_imodel_data_type * data, 
                                     double nsc) {
  

  int nstate        = matrix_get_rows( data->prior0 );
  int ens_size      = matrix_get_columns( data->prior0 );
  int nrmin         = util_int_min( ens_size , nstate); 
  matrix_type * Dm  = matrix_alloc_copy( data->prior0 );
  
  matrix_type * Um  = matrix_alloc( nstate , nrmin  );     /* Left singular vectors.  */
  matrix_type * VmT = matrix_alloc( nrmin , ens_size );    /* Right singular vectors. */
  double * Wm       = util_calloc( nrmin , sizeof * Wm ); 

 
   matrix_subtract_row_mean(Dm);


  //This routine only computes the SVD of Ensemble State matrix  

  
  for (int i=0; i<nstate; i++){
    double sc = nsc/ (data->Csc[i]);
    matrix_scale_row (Dm,i, sc);
  }


  int nsign1 = enkf_linalg_svd_truncation(Dm , data->truncation , -1 , DGESVD_MIN_RETURN  , Wm , Um , VmT);
  printf("The significant Eigen values are %d\n",nsign1);

  enkf_linalg_rml_enkfAm(Um, Wm, nsign1);
  data->Am=matrix_alloc_copy(Um);
  printf("\n Init1 completed\n");
  matrix_free(Um);
  matrix_free(VmT);
  matrix_free(Dm);
  free(Wm);
 
}

void  rml_enkf_imodel_Create_Csc(rml_enkf_imodel_data_type * data){
  // Create the scaling matrix based on the state vector

  int nstate        = matrix_get_rows( data->active_prior );
  int ens_size      = matrix_get_columns( data->active_prior );
 
  
 for (int i=0; i < nstate; i++) {
   double sumrow = matrix_get_row_sum(data->active_prior , i);
    double tmp = sumrow / ens_size;
    if (abs(tmp)< 1)
      data->Csc[i]=0.05;
    else
      data->Csc[i]= 1;
 }

}

void rml_enkf_imodel_scalingA(matrix_type *A, double * Csc, bool invert ){
  int nrows = matrix_get_rows(A);
  if (invert)
    for (int i=0; i< nrows ; i++)
      {
        double sc= 1/Csc[i];
        matrix_scale_row(A, i, sc);
      }
  else
    for (int i=0; i< nrows ; i++)
      {
        double sc= Csc[i];
        matrix_scale_row(A, i, sc);
      }
}




static void rml_enkf_imodel_initA__(rml_enkf_imodel_data_type * data, 
                                    matrix_type * A ,
                                    matrix_type * S , 
                                    matrix_type * Cd , 
                                    matrix_type * E , 
                                    matrix_type * D ,
                                    matrix_type * Udr,
                                    double * Wdr,
                                    matrix_type * VdTr) {

  int nrobs         = matrix_get_rows( S );
  int ens_size      = matrix_get_columns( S );
  double a = data->lambda + 1;
  matrix_type *tmp  = matrix_alloc (nrobs, ens_size);
  double nsc = 1/sqrt(ens_size-1);

  
  printf("The lamda Value is %5.5f\n",data->lambda);
  printf("The Value of Truncation is %4.2f \n",data->truncation);

  matrix_subtract_row_mean( S );           /* Shift away the mean in the ensemble predictions*/
  matrix_inplace_diag_sqrt(Cd);
  matrix_dgemm(tmp, Cd, S,false, false, 1.0, 0.0);
  matrix_scale(tmp, nsc);
  
  printf("The Scaling of data matrix completed !\n ");


  // SVD(S)  = Ud * Wd * Vd(T)
  int nsign = enkf_linalg_svd_truncation(tmp , data->truncation , -1 , DGESVD_MIN_RETURN  , Wdr , Udr , VdTr);
  
  /* After this we only work with the reduced dimension matrices */
  
  printf("The number of siginificant ensembles are %d \n ",nsign);
  
  matrix_type * X1   = matrix_alloc( nsign, ens_size);
  matrix_type * X2    = matrix_alloc (nsign, ens_size );
  matrix_type * X3    = matrix_alloc (ens_size, ens_size );
  
  
  // Compute the matrices X1,X2,X3 and dA 
  enkf_linalg_rml_enkfX1(X1, Udr ,D ,Cd );  //X1 = Ud(T)*Cd(-1/2)*D   -- D= -(dk-d0)
  enkf_linalg_rml_enkfX2(X2, Wdr ,X1 ,a, nsign);  //X2 = ((a*Ipd)+Wd^2)^-1  * X1

  matrix_free(X1);

  enkf_linalg_rml_enkfX3(X3, VdTr ,Wdr,X2, nsign);  //X3 = Vd *Wd*X2
  printf("The X3 matrix is computed !\n ");

  matrix_type *dA1= matrix_alloc (matrix_get_rows(A), ens_size);
  matrix_type * Dm  = matrix_alloc_copy( A );

  matrix_subtract_row_mean( Dm );      /* Remove the mean from the ensemble of model parameters*/
  matrix_scale(Dm, nsc);

  enkf_linalg_rml_enkfdA(dA1, Dm, X3);      //dA = Dm * X3   
  matrix_inplace_add(A,dA1); //dA 

  matrix_free(X3);
  matrix_free(Dm);
  matrix_free(dA1);
}


void rml_enkf_imodel_init2__( rml_enkf_imodel_data_type * data,
                              matrix_type *A,
                              matrix_type *Acopy,
                              double * Wdr,
                              double nsc,
                              matrix_type * VdTr) {


  int nstate        = matrix_get_rows( Acopy );
  int ens_size      = matrix_get_columns( Acopy );
  matrix_type * Dk  = matrix_alloc_copy( Acopy );

  double a = data->lambda + 1;
  matrix_type *Am= matrix_alloc_copy(data->Am);
  matrix_type *Apr= matrix_alloc_copy(data->active_prior);
  double *Csc = util_calloc(nstate , sizeof * Csc ); 
  for (int i=0; i< nstate ; i++)
    {
      Csc[i]= data->Csc[i];
    }
  int nsign1= matrix_get_columns(data->Am);
  

  matrix_type * X4  = matrix_alloc(nsign1,ens_size);
  matrix_type * X5  = matrix_alloc(nstate,ens_size);
  matrix_type * X6  = matrix_alloc(ens_size,ens_size);
  matrix_type * X7  = matrix_alloc(ens_size,ens_size);
  matrix_type * dA2 = matrix_alloc(nstate, ens_size);
  

  //Compute dA2 
  printf("\n Starting init 2 \n");
  matrix_inplace_sub(Dk, Apr);
  rml_enkf_imodel_scalingA(Dk,Csc,true);

  enkf_linalg_rml_enkfX4(X4, Am, Dk);
  matrix_free(Dk);

  enkf_linalg_rml_enkfX5(X5, Am, X4);
  printf("\nMatrix X5 computed\n");

  matrix_type * Dk1  = matrix_alloc_copy( Acopy );
  matrix_subtract_row_mean(Dk1);
  rml_enkf_imodel_scalingA(Dk1,Csc,true);
  matrix_scale(Dk1,nsc);

  enkf_linalg_rml_enkfX6(X6, Dk1,X5);
  printf("Matrix X6 computed!\n");

  enkf_linalg_rml_enkfX7(X7, VdTr ,Wdr, a, X6);
  printf("Matrix X7 computed!\n");

  rml_enkf_imodel_scalingA(Dk1,Csc,false);
  printf("Matrix Dk1 Scaling done!\n");

  enkf_linalg_rml_enkfXdA2(dA2,Dk1,X7);
  printf("Matrix dA2 computed!\n");

  matrix_inplace_sub(A, dA2);

  free(Csc);
  matrix_free(Am);
  matrix_free(Apr);
  matrix_free(X4); 
  matrix_free(X5);
  matrix_free(X6);
  matrix_free(X7);
  matrix_free(dA2);
  matrix_free(Dk1);

    
}


static void rml_enkf_imodel_updateA_iter0(rml_enkf_imodel_data_type * data,
                                          matrix_type * A , 
                                          matrix_type * S , 
                                          matrix_type * R , 
                                          matrix_type * dObs , 
                                          matrix_type * E , 
                                          matrix_type * D,
                                          matrix_type * Cd) {
        
  matrix_type * Skm = matrix_alloc(matrix_get_columns(D),matrix_get_columns(D));
  int ens_size      = matrix_get_columns( S );
  int nrobs         = matrix_get_rows( S );
  int nrmin         = util_int_min( ens_size , nrobs); 
  int state_size    = matrix_get_rows( A );
  matrix_type * Ud  = matrix_alloc( nrobs , nrmin    );    /* Left singular vectors.  */
  matrix_type * VdT = matrix_alloc( nrmin , ens_size );    /* Right singular vectors. */
  double * Wd       = util_calloc( nrmin , sizeof * Wd ); 
  double nsc = 1/sqrt(ens_size-1); 

  
  data->Sk  = enkf_linalg_data_mismatch(D,Cd,Skm);  
  data->Std = matrix_diag_std(Skm,data->Sk);
  
  if (data->lambda0 < 0)
    data->lambda = pow(10 , floor(log10(data->Sk/(2*nrobs))) );
  else
    data->lambda = data->lambda0;
  
  rml_enkf_common_store_state( data->state  , A , data->ens_mask );
  rml_enkf_common_store_state( data->prior0 , A , data->ens_mask );
  
  
  data->Csc     = util_calloc(state_size , sizeof * data->Csc);
  rml_enkf_imodel_Create_Csc( data );
  rml_enkf_imodel_initA__(data , A, S , Cd , E , D , Ud , Wd , VdT);
  rml_enkf_imodel_init1__(data , nsc);
  
  /*
    printf("Prior Objective function value is %5.3f \n", data->Sk);
    fprintf(fp,"Iteration number\t   Lambda Value \t    Current Mean (OB FN) \t    Old Mean\t     Current Stddev\n");
    fprintf(fp, "\n\n");
    fprintf(fp,"%d     \t\t       NA       \t      %5.5f      \t         \t   %5.5f    \n",data->iteration_nr, data->Sk, data->Std);
  */

  matrix_free( Skm );
  matrix_free( Ud );
  matrix_free( VdT );
  free( Wd );
}





void rml_enkf_imodel_updateA(void * module_data , 
                      matrix_type * A , 
                      matrix_type * S , 
                      matrix_type * R , 
                      matrix_type * dObs , 
                      matrix_type * E , 
                      matrix_type * D) {


  rml_enkf_imodel_data_type * data = rml_enkf_imodel_data_safe_cast( module_data );
  double Sk_new;
  double  Std_new;
  int nrobs         = matrix_get_rows( S );
  int ens_size      = matrix_get_columns( S );
  double nsc        = 1/sqrt(ens_size-1); 
  FILE *fp = util_fopen("rml_enkf_imodel_output","a");
  matrix_type * Cd  = matrix_alloc( nrobs, nrobs );
 
  
  enkf_linalg_Covariance(Cd ,E ,nsc, nrobs);
  matrix_inv(Cd);

  
  if (data->iteration_nr == 0) {
    rml_enkf_imodel_updateA_iter0(data , A , S , R , dObs , E , D , Cd);
  } else {
    int nrmin         = util_int_min( ens_size , nrobs); 
    matrix_type * Ud  = matrix_alloc( nrobs , nrmin    );    /* Left singular vectors.  */
    matrix_type * VdT = matrix_alloc( nrmin , ens_size );    /* Right singular vectors. */
    double * Wd       = util_calloc( nrmin , sizeof * Wd ); 
    matrix_type * Skm = matrix_alloc(matrix_get_columns(D),matrix_get_columns(D));
    matrix_type * Acopy  = matrix_alloc_copy (A);
    Sk_new = enkf_linalg_data_mismatch(D,Cd,Skm);  //Calculate the intitial data mismatch term
    Std_new = matrix_diag_std(Skm,Sk_new);
    
    printf(" Current Objective function value is %5.3f \n\n",Sk_new);
    printf(" The old Objective function value is %5.3f \n", data->Sk);
    {
      bool mismatch_reduced = false;
      bool std_reduced = false;

      if (Sk_new < data->Sk)
        mismatch_reduced = true;
      
      if (Std_new <= data->Std)
        std_reduced = true;
      
      fprintf(fp,"%d     \t\t      %5.5f      \t      %5.5f      \t    %5.5f    \t   %5.5f    \n",data->iteration_nr,data->lambda, Sk_new,data->Sk, Std_new);

      if (mismatch_reduced) {
        /*
          Stop check: if ( (1- (Sk_new/data->Sk)) < .0001)  // check convergence ** model change norm has to be added in this!!
        */
        if (std_reduced) 
          data->lambda = data->lambda * data->lambda_reduce_factor;

        rml_enkf_common_store_state(data->state , A , data->ens_mask );
        rml_enkf_common_recover_state( data->prior0 , data->active_prior , data->ens_mask );
        
        rml_enkf_imodel_initA__(data , A , S , Cd , E , D , Ud , Wd , VdT);
        rml_enkf_imodel_init2__(data , A , Acopy , Wd , nsc , VdT);

        data->Sk = Sk_new;
        data->Std=Std_new;
        data->iteration_nr++;
      } else {
        data->lambda = data->lambda * data->lambda_increase_factor;
        
        rml_enkf_common_recover_state( data->state , A , data->ens_mask );
        rml_enkf_common_recover_state( data->prior0 , data->active_prior , data->ens_mask );
        
        rml_enkf_imodel_initA__(data , A , S , Cd , E , D , Ud , Wd , VdT);
        rml_enkf_imodel_init2__(data , A , Acopy , Wd , nsc , VdT);
      }
    }
    matrix_free(Acopy);
    matrix_free(Skm);
    matrix_free( Ud );
    matrix_free( VdT );
    free( Wd );
  }

  //setting the lower bound for lambda
  if (data->lambda <.01)
    data->lambda= .01;

  printf ("The current iteration number is %d \n ", data->iteration_nr);
  
  matrix_free(Cd);
  fclose(fp);
}




void rml_enkf_imodel_init_update(void * arg , 
                                 const bool_vector_type * ens_mask , 
                                 const matrix_type * S , 
                                 const matrix_type * R , 
                                 const matrix_type * dObs , 
                                 const matrix_type * E , 
                                 const matrix_type * D ) {
  
  rml_enkf_imodel_data_type * module_data = rml_enkf_imodel_data_safe_cast( arg );
  bool_vector_memcpy( module_data->ens_mask , ens_mask );
}



bool rml_enkf_imodel_set_double( void * arg , const char * var_name , double value) {
  rml_enkf_imodel_data_type * module_data = rml_enkf_imodel_data_safe_cast( arg );
  {
    bool name_recognized = true;

    if (strcmp( var_name , ENKF_TRUNCATION_KEY_) == 0)
      rml_enkf_imodel_set_truncation( module_data , value );
    if (strcmp( var_name , LAMBDA_INCREASE_FACTOR_KEY) == 0)
      rml_enkf_imodel_set_lambda_increase_factor( module_data , value );
    if (strcmp( var_name , LAMBDA_REDUCE_FACTOR_KEY) == 0)
      rml_enkf_imodel_set_lambda_reduce_factor( module_data , value );
    if (strcmp( var_name , LAMBDA0_KEY) == 0)
      rml_enkf_imodel_set_lambda0( module_data , value );
    else
      name_recognized = false;

    return name_recognized;
  }
}


bool rml_enkf_imodel_set_int( void * arg , const char * var_name , int value) {
  rml_enkf_imodel_data_type * module_data = rml_enkf_imodel_data_safe_cast( arg );
  {
    bool name_recognized = true;
    
    if (strcmp( var_name , ENKF_NCOMP_KEY_) == 0)
      rml_enkf_imodel_set_subspace_dimension( module_data , value );
    if (strcmp( var_name , ITER_KEY) == 0)
      rml_enkf_imodel_set_iteration_nr( module_data , value );
    else
      name_recognized = false;

    return name_recognized;
  }
}


bool rml_enkf_imodel_set_bool( void * arg , const char * var_name , bool value) {
  rml_enkf_imodel_data_type * module_data = rml_enkf_imodel_data_safe_cast( arg );
  {
    bool name_recognized = true;
    
    if (strcmp( var_name , USE_PRIOR_KEY) == 0)
      rml_enkf_imodel_set_use_prior( module_data , value );
    else
      name_recognized = false;

    return name_recognized;
  }
}


long rml_enkf_imodel_get_options( void * arg , long flag ) {
  rml_enkf_imodel_data_type * module_data = rml_enkf_imodel_data_safe_cast( arg );
  {
    return module_data->option_flags;
  }
}



 bool rml_enkf_imodel_has_var( const void * arg, const char * var_name) {
   {
     if (strcmp(var_name , ITER_KEY) == 0)
       return true;
     else if (strcmp(var_name , USE_PRIOR_KEY) == 0)
       return true;
     else if (strcmp(var_name , LAMBDA_INCREASE_FACTOR_KEY) == 0)
       return true;
     else if (strcmp(var_name , LAMBDA_REDUCE_FACTOR_KEY) == 0)
       return true;
     else if (strcmp(var_name , LAMBDA0_KEY) == 0)
       return true;
     else if (strcmp(var_name , ENKF_TRUNCATION_KEY_) == 0)
       return true;
     else
       return false;
   }
 }



 
 int rml_enkf_imodel_get_int( const void * arg, const char * var_name) {
   const rml_enkf_imodel_data_type * module_data = rml_enkf_imodel_data_safe_cast_const( arg );
   {
     if (strcmp(var_name , ITER_KEY) == 0)
       return module_data->iteration_nr;
     else
       return -1;
   }
 }


 bool rml_enkf_imodel_get_bool( const void * arg, const char * var_name) {
   const rml_enkf_imodel_data_type * module_data = rml_enkf_imodel_data_safe_cast_const( arg );
   {
     if (strcmp(var_name , USE_PRIOR_KEY) == 0)
       return module_data->use_prior;
     else
       return false;
   }
 }



 double rml_enkf_imodel_get_double( const void * arg, const char * var_name) {
   const rml_enkf_imodel_data_type * module_data = rml_enkf_imodel_data_safe_cast_const( arg );
   {
     if (strcmp(var_name , LAMBDA_REDUCE_FACTOR_KEY) == 0)
       return module_data->lambda_reduce_factor;
     if (strcmp(var_name , LAMBDA_INCREASE_FACTOR_KEY) == 0)
       return module_data->lambda_increase_factor;
     if (strcmp(var_name , LAMBDA0_KEY) == 0)
       return module_data->lambda0;
     if (strcmp(var_name , ENKF_TRUNCATION_KEY_) == 0)
       return module_data->truncation;
     else
       return -1;
   }
 }






#ifdef INTERNAL_LINK
#define SYMBOL_TABLE rml_enkf_imodel_symbol_table
#else
#define SYMBOL_TABLE EXTERNAL_MODULE_SYMBOL
#endif


analysis_table_type SYMBOL_TABLE = {
    .alloc           = rml_enkf_imodel_data_alloc,
    .freef           = rml_enkf_imodel_data_free,
    .set_int         = rml_enkf_imodel_set_int , 
    .set_double      = rml_enkf_imodel_set_double , 
    .set_bool        = rml_enkf_imodel_set_bool, 
    .set_string      = NULL , 
    .get_options     = rml_enkf_imodel_get_options , 
    .initX           = NULL,
    .updateA         = rml_enkf_imodel_updateA ,  
    .init_update     = rml_enkf_imodel_init_update ,
    .complete_update = NULL,
    .has_var         = rml_enkf_imodel_has_var,
    .get_int         = rml_enkf_imodel_get_int,
    .get_double      = rml_enkf_imodel_get_double,
    .get_bool        = rml_enkf_imodel_get_bool,
    .get_ptr         = NULL, 
};

