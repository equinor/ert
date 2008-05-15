#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <util.h>
#include "blas.h"
#include "cost_func.h"
#include "pre-image.h"

/********************************************************************/

void pre_image_approx(const cost_func_type *cost_func,
                      const int ntries,
                      const int n, 
                      const int ny, 
                      const double **y,
                      const double *alpha,
                      const double lambda,
                      const double *mu,
                      const double *low_bnd, 
                      const double *high_bnd,
                      const int *nbd,
                      double *x
                      )
{
  const int one = 1;
  const double dmone = -1.0;

  bool converged;
  int i,j,iprint,m;
  double f,factr,pgtol,tsq,aka;

  int *lsave;
  char *csave,*task;
  int *isave,*iwa;
  double *dsave,*g,*wa,*xm;

  /******************************************************************/

  // Configure L-BFGS-B
  // factr = 1.0E12;
  // pgtol = 1.0E-2;
  factr = 1.0E2;
  pgtol = 1.0E-5;
  m = 4;

  /* The L-BFGS-B fortran code doesn't handle its own memory. We
     need to allocate the memory here...
  */

  lsave =   util_malloc(4                     *sizeof *lsave, __func__);
  csave =   util_malloc(60                    *sizeof *csave, __func__);
  task  =   util_malloc(60                    *sizeof *task,  __func__);
  iwa   =   util_malloc(3*n                   *sizeof *iwa,   __func__);
  isave =   util_malloc(44                    *sizeof *isave, __func__);
  g     =   util_malloc(n                     *sizeof *g,     __func__);
  dsave =   util_malloc(29                    *sizeof *dsave, __func__);
  wa    =   util_malloc(((2*m+4)*n+11*m*m+8*m)*sizeof *wa,    __func__); 
  xm    =   util_malloc(n                     *sizeof *xm,    __func__);

  /* Don't be verbose unless debugging is enabled.
  */
  iprint = -1;
  #ifdef DEBUG
  iprint = 1;
  printf("Verbose mode enabled.\n");
  #endif


  /* Suppose that we want to find an approximative x such that
     
      phi(x) = \sum \alpha_i phi(y_i) 

     for given alpha and y. Using the cost function method implemented in
     this library, we need to calculate the sum

        aka = \sum_i \sum_j \alpha_i\alpha_j <phi(y_i),phi(y_j)>

     where <,> is a dot product. This is computationally expensive, so
     we only do this ONCE and store it in aka for the iterations.
  */
  aka = kernel_featurespace_dist_get_aka(cost_func->kernel_list,n,ny,alpha,y);

  #ifdef DEBUG
  printf("aka: %f\n",aka);
  #endif

  /* If the convergence criteria in L-BFGS-B are met or
     squared distance in pre-image space goes negative
     (i.e., rouding error dominates), converged is set
     to true.
  */
  converged = false;
  
  for(i=0; i < ntries; i++)
  {
    /* Tell the L-BFGS-B routine that we are to start
       a new minimization and initialize it in a random
       sample from {y_1,y_2,...,y_n}
    */

    util_memcpy_string_C2f90("START",task,60);
    j = rand() % ny;
    dcopy_(&n,y[j],&one,x,&one);
    #ifdef DEBUG
    printf("Starting L-BFGS-B in y[%i]..\n",j);
    #endif

    do  
    {
      // Call the L-BFGS-B routine
      setulb_(&n,&m,x,low_bnd,high_bnd,nbd,&f,g,
              &factr,&pgtol,wa,iwa,task,&iprint,csave,lsave,isave,dsave);
      if(strncmp(task,"FG",2) == 0)
      {
        tsq = kernel_featurespace_dist_squared(cost_func->kernel_list,n,x,ny,alpha,(const double**) y,&aka);
        #ifdef DEBUG
        printf("tsq: %f\n",tsq);
        #endif

        /* tsq is the squared distance in the feature space,
           if it goes negative, rouding error dominates and
           we have convergence (or a bogus result if the
           scaling is off!)
        */
        if(tsq < 0.0)
        {
          #ifdef DEBUG
          printf("Iteration ended, tsq <0, calculations dominated by rounding error.\n");
          #endif

          util_memcpy_string_C2f90("CONV",task,60); 
          break;
        }
        
        /* If tsq is not negative, we calculate the 
           value and gradient of the cost function at
           the new iterate.
        */
        f = cost_func_apply(cost_func,tsq,n,lambda,x,mu);
        cost_func_apply_gradx(cost_func,tsq,n,lambda,x,mu,ny,alpha,y,g);
      }
    }
    while(strncmp(task,"FG",2)==0 || strncmp(task,"NEW_X",5)==0);

    /* The iteration has terminated.

       With some kernels (even ordered dp's in particular)
       the L-BFGS-B has a tendency to end the iterations in
       -x, when the true minima in is in x. To avoid this
       problem, we check the value of the cost function
       in -x. If it is less, we start a new iteration there.
    */
    #ifdef DEBUG
    printf("Iteration terminated, checking for mirroring.\n");
    #endif
    dcopy_(&n,x,&one,xm,&one);
    dscal_(&n,&dmone,xm,&one);

    tsq = kernel_featurespace_dist_squared(cost_func->kernel_list,n,xm,ny,alpha,y,&aka);
    if(cost_func_apply(cost_func,tsq,n,lambda,x,mu) < f)
    {
      /* The mirror problem has occured, we start
         a new minimization in -x.
      */
      #ifdef DEBUG
      printf("Mirror problem observed. Starting new iterations in -x\n");  
      #endif
      dscal_(&n,&dmone,x,&one);
      util_memcpy_string_C2f90("START",task,60);
      i--;
    }

    /* If L-BFGS-B met its internal convergence
       criteria, or tsq went negative, we have
       convergence and exit. If not, we do a new
       iteration in the outer for loop.
    */
    if(strncmp(task,"CONV",4) == 0)
    {
      #ifdef DEBUG
      printf("Iteration terminated, convergence.\n");
      #endif
      converged = true;
      break;
    }
  }

  /* The pre-image solver has terminated. 

     Check for convergence. If it failed, terminate
     the program to avoid bogus results.
  */
  if(converged)
  {
    tsq = kernel_featurespace_dist_squared(cost_func->kernel_list,n,x,ny,alpha,y,&aka);
    if(tsq < 0.0)
    {
      tsq = 0.0;
    }
    printf("Pre-image solver converged. tau: %f\n",sqrt(tsq));
  }
  else{
    fprintf(stderr,"%s: Pre-image solver did not converge -- aborting.\n",__func__);
    abort();
  }

  // Free workingspace
  free(lsave);
  free(csave);
  free(task);
  free(iwa);
  free(isave);
  free(g);
  free(dsave);
  free(wa);
  free(xm);

};

/*****************************************************************

  Wrappers for some common cost functions, for use with fortran

*****************************************************************/
  
void fw_pre_image_approx_dot_xpsqx_(
                            const int *ntries,
                            const int *n,
                            const int *ny,
                            const double *y,
                            const double *alpha,
                            const double *low_bnd,
                            const double *high_bnd,
                            const int *nbd,
                            double *x
                            )
{
  // Configure cost function 
  f_func f = {X};
  reg_func reg = {NONE};
  
  cost_func_type *cost_func;
  kernel_list_type *kernel_dot;

  kernel_type kernels[1] = {DOT};
  double weights[1] = {1.0};
  double params[1] = {1.0};

  // Wrapper for y
  const double **wrap_y;

  // This is unused, since we are not using regularization
  double *mu;

  int i;
  /*********************************************************/

  // Allocate kernel and cost function
  kernel_dot = kernel_list_alloc(1,kernels,weights,params);
  cost_func = cost_func_alloc(kernel_dot,f,reg); 

  // Wrap y
  wrap_y = util_malloc(*n *sizeof *wrap_y,__func__);
  wrap_y[0] = y;
  for(i=1; i< *ny; i++)
  {
    wrap_y[i]  = wrap_y[i-1] + *n;
  }


  // Get pre-image approximation
  pre_image_approx(cost_func,*ntries,*n,*ny, (const double **) wrap_y,alpha,0.0,mu,low_bnd,high_bnd,nbd,x);

  // Clean up
  kernel_list_free(kernel_dot);
  cost_func_free(cost_func);
  free(wrap_y);
};



void fw_pre_image_approx_2xdot_xpsqx_(
                            const int *ntries,
                            const int *n,
                            const int *ny,
                            const double *y,
                            const double *alpha,
                            const double *low_bnd,
                            const double *high_bnd,
                            const int *nbd,
                            double *x
                            )
{
  // Configure cost function 
  f_func f = {XPLUSSQRTX};
  reg_func reg = {NONE};
  
  cost_func_type *cost_func;
  kernel_list_type *kernel_2xdot;

  kernel_type kernels[2] = {DOT,DOT};
  double weights[2] = {1.0,1.0};
  double params[2] = {1.0,2.0};

  // Wrapper for y
  const double **wrap_y;

  // This is unused, since we are not using regularization
  double *mu;

  int i;
  /*********************************************************/

  // Allocate kernel and cost function
  kernel_2xdot = kernel_list_alloc(2,kernels,weights,params);
  cost_func = cost_func_alloc(kernel_2xdot,f,reg); 

  // Wrap y
  wrap_y = util_malloc(*n *sizeof *wrap_y,__func__);
  wrap_y[0] = y;
  for(i=1; i< *ny; i++)
  {
    wrap_y[i]  = wrap_y[i-1] + *n;
  }


  // Get pre-image approximation
  pre_image_approx(cost_func,*ntries,*n,*ny, (const double **) wrap_y,alpha,0.0,mu,low_bnd,high_bnd,nbd,x);

  // Clean up
  kernel_list_free(kernel_2xdot);
  cost_func_free(cost_func);
  free(wrap_y);
};



void fw_pre_image_approx_3xdot_xpsqx_(
                            const int *ntries,
                            const int *n,
                            const int *ny,
                            const double *y,
                            const double *alpha,
                            const double *low_bnd,
                            const double *high_bnd,
                            const int *nbd,
                            double *x
                            )
{
  // Configure cost function 
  f_func f = {XPLUSSQRTX};
  reg_func reg = {NONE};
  
  cost_func_type *cost_func;
  kernel_list_type *kernel_3xdot;

  kernel_type kernels[3] = {DOT,DOT,DOT};
  double weights[3] = {1.0,1.0,1.0};
  double params[3] = {1.0,2.0,3.0};

  // Wrapper for y
  const double **wrap_y;

  // This is unused, since we are not using regularization
  double *mu;

  int i;
  /*********************************************************/

  // Allocate kernel and cost function
  kernel_3xdot = kernel_list_alloc(3,kernels,weights,params);
  cost_func = cost_func_alloc(kernel_3xdot,f,reg); 

  // Wrap y
  wrap_y = util_malloc(*n *sizeof *wrap_y,__func__);
  wrap_y[0] = y;
  for(i=1; i< *ny; i++)
  {
    wrap_y[i]  = wrap_y[i-1] + *n;
  }


  // Get pre-image approximation
  pre_image_approx(cost_func,*ntries,*n,*ny, (const double **) wrap_y,alpha,0.0,mu,low_bnd,high_bnd,nbd,x);

  // Clean up
  kernel_list_free(kernel_3xdot);
  cost_func_free(cost_func);
  free(wrap_y);
};



void fw_pre_image_approx_gauss_xpsqx_(
                            const int *ntries,
                            const int *n,
                            const int *ny,
                            const double *y,
                            const double *alpha,
                            const double *low_bnd,
                            const double *high_bnd,
                            const int *nbd,
                            double *x
                            )
{
  // Configure cost function 
  f_func f = {XPLUSSQRTX};
  reg_func reg = {NONE};
  
  cost_func_type *cost_func;
  kernel_list_type *kernel_gauss;

  kernel_type kernels[1] = {GAUSS};
  double weights[1] = {(double) *n};
  double params[1] = {(double) *n};

  // Wrapper for y
  const double **wrap_y;

  // This is unused, since we are not using regularization
  double *mu;

  int i;
  /*********************************************************/

  // Allocate kernel and cost function
  kernel_gauss = kernel_list_alloc(1,kernels,weights,params);
  cost_func = cost_func_alloc(kernel_gauss,f,reg); 

  // Wrap y to c style
  wrap_y = util_malloc(*n *sizeof *wrap_y,__func__);
  wrap_y[0] = y;
  for(i=1; i< *ny; i++)
  {
    wrap_y[i]  = wrap_y[i-1] + *n;
  }


  // Get pre-image approximation
  pre_image_approx(cost_func,*ntries,*n,*ny, (const double **) wrap_y,alpha,0.0,mu,low_bnd,high_bnd,nbd,x);

  // Clean up
  kernel_list_free(kernel_gauss);
  cost_func_free(cost_func);
  free(wrap_y);
};
