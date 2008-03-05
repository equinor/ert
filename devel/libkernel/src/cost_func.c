#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <util.h>
#include "kernel_core.h"
#include "cost_func.h"
#include "blas.h"

/********************************************************************/



/** \brief Allocator for the cost_func_type.
  *
  * Given a kernel_list_type, a strictly increasing function f and a regularization function
  * reg, a cost_func_type is function
  *
  * cost_func = f( tausq(x) ) + reg(x,mu,lambda)
  *
  * where 
  *
  * tausq(x) = dist(\phi(x) , \sum_i \alpha_i \phi(y_i))^2.
  */
cost_func_type *cost_func_alloc(kernel_list_type *kernel_list,f_func f_name,reg_func reg_name)
{
  cost_func_type *cost_func = util_malloc(sizeof *cost_func,__func__);
  cost_func->kernel_list = kernel_list;
  if(f_name == X)
  {
    cost_func->f = f_func_x;
    cost_func->df = f_func_dx;
  }
  else if(f_name == SQRTX)
  {
    cost_func->f = f_func_sqrtx;
    cost_func->df = f_func_dsqrtx;
  }
  else if(f_name == XPLUSSQRTX)
  {
    cost_func->f = f_func_xplussqrtx;
    cost_func->df = f_func_dxplussqrtx;
  }
  else
  {
    fprintf(stderr,"%s: unkown f_func - aborting \n",__func__);
    abort();
  }
  if(reg_name == NONE)
  {
    cost_func->reg = reg_func_none;
    cost_func->gradx_reg = reg_func_none_gradx;
  }
  else
  {
    fprintf(stderr,"%s: unkown reg_func - aborting \n",__func__);
    abort();
  }
  return cost_func;
};



/** \brief Deallocator for the cost_func_type.
  */
void cost_func_free(cost_func_type *cost_func)
{
  free(cost_func);
};



/** \brief Calculate the value of a cost_func_type at (tausquared,lambda,x,mu).
  */
double cost_func_apply(const cost_func_type *cost_func, const double tausquared,const int n,const double lambda, const double *x,
                       const double *mu)
{
  if(tausquared < 0.0)
  {
    fprintf(stderr,"%s: trying to evaluate cost function with tsq < 0.0 - aborting\n",__func__);
    abort();
  }
  return cost_func->f(tausquared) + cost_func->reg(n,lambda,x,mu);
};



/** \brief Calculate the gradient of a cost_func_type at (tausquared,lambda,x,mu).
  */
void cost_func_apply_gradx(const cost_func_type *cost_func,const double tausquared, const int n, const double lambda, const double *x,
                           const double *mu,const int ny,const double *alpha,const double **y, double *g)
{
  const int one = 1;
  const double dzero = 0.0;
  const double done = 1.0;
  int i;
  double beta;
  double gamma;
  double *dg;

  if(tausquared < 0.0)
  {
    fprintf(stderr,"%s: trying to evaluate grad of cost function with tsq < 0.0 - aborting\n",__func__);
    abort();
  }
  
  gamma = cost_func->df(tausquared);
  dscal_(&n,&dzero,g,&one);
  kernel_list_apply_gradxx(cost_func->kernel_list,n,x,g);

  dg = util_malloc(n*sizeof *dg,__func__);
  for(i=0; i<ny; i++)
  {
    kernel_list_apply_gradx(cost_func->kernel_list,n,x,y[i],dg);
    beta = -2.0 * alpha[i];
    daxpy_(&n,&beta,dg,&one,g,&one);
  }

  cost_func->gradx_reg(n,lambda,x,mu,dg);
  daxpy_(&n,&done,dg,&one,g,&one);
  free(dg);
};



double f_func_x(const double x)
{
  return x;  
};



double f_func_dx(const double x)
{
  return 1.0;
};



double f_func_sqrtx(const double x)
{
  return sqrt(x);
};



double f_func_dsqrtx(const double x)
{
  // We account for rounding errors
  if( x < 1E-10)
  { 
    return 0.0;
  }
  else
  {
    return 0.5 / sqrt(x);
  }
};



double f_func_xplussqrtx(const double x)
{
  return sqrt(x) + x;
};



double f_func_dxplussqrtx(const double x)
{
  // We account for rounding errors
  if( x < 1E-10)
  { 
    return 1.0;
  }
  else
  {
    return 0.5 / sqrt(x) + 1.0;
  }
};



double reg_func_none(const int n,const double lamdba,const double *x,const double *mu)
{
  return 0.0;
};



void reg_func_none_gradx(const int n, const double lambda, const double *x, const double *mu, double *g)
{
  const int one = 1;
  const double dzero = 0.0;

  dscal_(&n,&dzero,g,&one);
};
