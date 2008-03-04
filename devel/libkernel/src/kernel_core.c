/** \file kernel_core.c
  * 
  * \breif Core functions and structs for libkernel
  *
  * Background information
  *
  * A kernel k on R^n \times R^n is a symmetric, semidefinite function
  * into R. Obviously, if k is a kernel, then \alpha \cdot k is also
  * a kernel whenever \alpha > 0. Furthermore, if k_1 and k_2 are kernels,
  * so is k_1 + k_2. By induction, given a sequence of positive scalars
  * \alpha_i and a sequence of kernels k_i, then k = \sum_i \alpha_i\cdot k_i
  * is a valid kernel.
  *
  * Purpose of this file
  *
  * The purpose of this file is to provide the neccesary facilities
  * to build a kernel k from a weighted sum of simpler kernels. Furthermore,
  * it provides facilities to evaluate k(x,y), \grad_x k(x,y) and \grad_x k(x,x)
  *
  * @author Jon Gustav Vabø
  */

#include <math.h>
#include <stdlib.h>
#include <util.h>
#include "blas.h"
#include "kernel_core.h"

/** \brief Functional form of the dot product kernel
  *
  * The dot product kernel of order d is defined by k(x,y) = (x^T y)^d
  * 
  * For it to be a valid kernel, order should be a natural number. This is not enforced.
  */
double kernel_dotproduct(const int n,const double *x, const double *y, const double order)
{
  double k;
  const int one = 1;

  k = ddot_(&n,x,&one,y,&one);
  k = pow(k,order);

  return k;
};



/** \brief Gradient of the dot product kernel when k(x,y)
  *
  * Since k(x,y) = (x^T y) ^d = (\sum_i x_i y_i)^d, obviosuly
  * \delta k(x,y) \delta x_i = d (x^T y)^{d-1} y_i. Thus
  * \nabla k(x,y) = d (x^Ty)^{d-1} y
  */
void kernel_dotproduct_gradx(const int n, const double *x, const double *y,const double order, double *g)
{
  double alpha;
  const int one = 1;

  alpha = ddot_(&n,x,&one,y,&one);
  alpha = order * pow(alpha,order - 1);
  dcopy_(&n,y,&one,g,&one);
  dscal_(&n,&alpha,g,&one);
};



/** \brief Gradient of the dot product kernel when k(x,x)
  * 
  * Now, k(x,x) = (x^T x)^d, so \nabla k(x,x) = 2d(x^T x)^{d-1} x
  */
void kernel_dotproduct_gradxx(const int n, const double *x, const double order, double *g)
{
  double alpha;
  const int one = 1;

  alpha = ddot_(&n,x,&one,x,&one);
  alpha = 2.0 * order * pow(alpha,order-1);
  dcopy_(&n,x,&one,g,&one);
  dscal_(&n,&alpha,g,&one);
};



/** \brief Functional form of the Gaussian kernel
  *
  * The Gaussian kernel is defined as k(x,y) = \exp( - 1 / beta \|x-y\|^2)
  *
  * For it to be valid kernel, beta must be positive. This is enforced when using 
  * kernel_node_alloc and kernel_list_alloc, but not at evaluation.
  *
  * A value of beta \approx n yields best results.
  */
double kernel_gauss(const int n,const double *x,const double *y,const double beta)
{
  double k;
  double *xmy;
  const int one = 1;
  const double  dmone = -1.0;

  xmy = util_malloc(n * sizeof *xmy,__func__);
  dcopy_(&n,x,&one,xmy,&one);
  daxpy_(&n,&dmone,y,&one,xmy,&one);
  k = ddot_(&n,xmy,&one,xmy,&one);
  k = exp(-k/beta); 
  free(xmy); 
  return k;
};



/** \brief Gradient of the Gaussian kernel at k(x,y)
  *
  * \nabla k(x,y) = -2 / beta exp(- 1/beta \|x-y\|^2) * (x-y)
  */
void kernel_gauss_gradx(const int n,const double *x,const double *y,const double beta,double *g)
{
  double alpha;
  const int one = 1;
  const double dmone = -1.0;

  dcopy_(&n,x,&one,g,&one);
  daxpy_(&n,&dmone,y,&one,g,&one);
  alpha = ddot_(&n,g,&one,g,&one);
  alpha = -2.0 / beta * exp( -1.0 / beta * alpha);
  dscal_(&n,&alpha,g,&one);
};



/** \brief Gradient of the Gaussian kernel at k(x,x)
  *
  * k(x,x) = 1, so \nabla k(x,x) = 0
  */
void kernel_gauss_gradxx(const int n,const double *x,const double beta,double *g)
{
  const double dzero = 0.0;
  const int one = 1;

  dscal_(&n,&dzero,g,&one);
};



/*

  TANH is NOT a valid kernel!

double kernel_tanh(const int n,const double *x, const double *y, const double beta)
{
  double k;
  const int one = 1;

  k = ddot_(&n,x,&one,y,&one);
  k = tanh(1 / beta * k);

  return k;
};



void kernel_tanh_gradx(const int n, const double *x, const double *y,const double beta, double *g)
{
  double alpha;
  const int one = 1;

  alpha = ddot_(&n,x,&one,y,&one);
  alpha = pow(tanh(1 / beta * alpha),2.0);
  alpha = (1.0 - alpha) * 1 / beta; 
  dcopy_(&n,y,&one,g,&one);
  dscal_(&n,&alpha,g,&one);
};



void kernel_tanh_gradxx(const int n, const double *x, const double beta, double *g)
{
  double alpha;
  const int one = 1;

  alpha = ddot_(&n,x,&one,x,&one);
  alpha = pow(tanh(1 / beta * alpha),2.0);
  alpha = (1.0 - alpha) * 2 / beta; 
  dcopy_(&n,x,&one,g,&one);
  dscal_(&n,&alpha,g,&one);
};
*/


kernel_node_type * kernel_node_alloc(const kernel_type kernel_name , double weight , double param)
{
  kernel_node_type *kernel = util_malloc(sizeof * kernel,__func__);
  kernel->weight = weight;
  kernel->param  = param;

  if (kernel_name == DOT)
  {
    kernel->func = kernel_dotproduct;
    kernel->gradx = kernel_dotproduct_gradx;
    kernel->gradxx = kernel_dotproduct_gradxx;
  }
  else if (kernel_name == GAUSS)
  { 
    kernel->func = kernel_gauss;
    kernel->gradx = kernel_gauss_gradx;
    kernel->gradxx = kernel_gauss_gradxx;
  }
  /*

    TANH is NOT a valid kernel
  else if (kernel_name == TANH)
  {
    kernel->func = kernel_tanh;
    kernel->gradx = kernel_tanh_gradx;
    kernel->gradxx = kernel_tanh_gradxx;
  }
  */
  else
  {
    fprintf(stderr,"%s: unknown kernel_type - aborting \n",__func__ );
    abort();
  }
  return kernel;
};



void kernel_node_free(kernel_node_type *kernel)
{
  free(kernel);
};



void kernel_node_assert(const kernel_node_type *kernel)
{
  if(kernel->weight < 0.0)
  {
    fprintf(stderr,"%s: kernel_node has weight: %f < 0 - aborting \n",__func__,kernel->weight);
    abort();
  }
  else if(kernel->param < 0.0)
  {
    fprintf(stderr,"%s: kernel_node has param: %f <0 - aborting \n",__func__,kernel->param);
  }
};



double kernel_apply(const kernel_node_type *kernel, const int n, const double *x, const double *y) 
{
  return kernel->weight * kernel->func(n,x,y,kernel->param);
};



void kernel_apply_gradx(const kernel_node_type *kernel, const int n, const double *x, const double *y, double *g)
{
  const int one = 1;
  kernel->gradx(n,x,y,kernel->param,g);
  dscal_(&n,&kernel->weight,g,&one);
};



void kernel_apply_gradxx(const kernel_node_type *kernel,const int n,const double *x, double *g)
{
  const int one = 1;
  kernel->gradxx(n,x,kernel->param,g);
  dscal_(&n,&kernel->weight,g,&one);
};



kernel_list_type* kernel_list_alloc(const int nk, kernel_type *kernel_names,double *weights, double *params)
{
  int i;

  kernel_list_type *kernel_list = util_malloc(sizeof *kernel_list,__func__);
  kernel_list->nk = nk;
  kernel_list->kernel_nodes = util_malloc(nk*sizeof *kernel_list->kernel_nodes,__func__);

  for(i=0; i<nk; i++)
  {
    kernel_list->kernel_nodes[i] = kernel_node_alloc(kernel_names[i],weights[i],params[i]);
  }
  return kernel_list;

};



void kernel_list_free(kernel_list_type *kernel_list)
{
  int i;
  for(i=0; i <kernel_list->nk; i++)
  {
    if(kernel_list->kernel_nodes[i] != NULL)
    {
      kernel_node_free(kernel_list->kernel_nodes[i]);
    }
  }
  free(kernel_list->kernel_nodes);
  free(kernel_list);
};



void kernel_list_assert(const kernel_list_type *kernel_list)
{
  int i;
  if(kernel_list->nk <= 0){
    fprintf(stderr,"%s: number of kernels in kernel list is: %i - aborting \n",__func__,kernel_list->nk);
    abort();
  }
  for(i=0; i<kernel_list->nk; i++)
  {
    kernel_node_assert(kernel_list->kernel_nodes[i]);
  };
};



double kernel_list_apply(const kernel_list_type *kernel_list, const int  n, const double *x, const double *y)
{
  int i;
  double k;
  
  k=0.0;
  //#pragma omp parallel for private(k)
  for(i=0; i<kernel_list->nk; i++)
  {
    k = k + kernel_apply(kernel_list->kernel_nodes[i],n,x,y); 
  }
  return k;
};



void kernel_list_apply_gradx(const kernel_list_type *kernel_list, const int n, const double *x, const double *y, double *g)
{
  int i;
  double *dg;
  const int one = 1;
  const double dzero = 0.0;
  const double done = 1.0;

  dscal_(&n,&dzero,g,&one);

  //#pragma omp parallel for private(dg)
  dg = util_malloc(n*sizeof *dg,__func__);
  for(i=0; i<kernel_list->nk; i++)
  {
    kernel_apply_gradx(kernel_list->kernel_nodes[i],n,x,y,dg);
    daxpy_(&n,&done,dg,&one,g,&one); 
  }

  free(dg);
};



void kernel_list_apply_gradxx(const kernel_list_type *kernel_list, const int n, const double *x, double *g)
{
  int i;
  double *dg;
  const int one = 1;
  const double dzero = 0.0;
  const double done = 1.0;

  dscal_(&n,&dzero,g,&one);

  //#pragma omp parallel for private(dg)
  dg = util_malloc(n*sizeof *dg,__func__);
  for(i=0; i<kernel_list->nk; i++)
  {
    kernel_apply_gradxx(kernel_list->kernel_nodes[i],n,x,dg);
    daxpy_(&n,&done,dg,&one,g,&one); 
  }

  free(dg);
};



double kernel_featurespace_dist_squared(const kernel_list_type *kernel_list,const int n,const double *x,const int ny,const double *alpha,const double **y,double *aka)
{
  int i;
  double dist_sq;

  if(aka == NULL)
  {
    aka = util_malloc(sizeof *aka,__func__);
    *aka = kernel_featurespace_dist_get_aka(kernel_list,n,ny,alpha,y);
  }
  dist_sq = kernel_list_apply(kernel_list,n,x,x);
  for(i=0; i<ny; i++)
  {
    dist_sq = dist_sq - 2.0*alpha[i]*kernel_list_apply(kernel_list,n,x,y[i]);
  }
  dist_sq = dist_sq + *aka;
  return dist_sq;
};



double kernel_featurespace_dist_get_aka(const kernel_list_type *kernel_list,const int n, const int ny,const double *alpha,const double **y)
{
  double aka;
  int i,j;

  // Note: This code can be optimized if we enforce k(x,y) = k(y,x)
  aka = 0.0;
  for(i=0; i<ny; i++)
  {
    for(j=0; j<ny; j++)
    {
      aka = aka + alpha[i]*alpha[j] * kernel_list_apply(kernel_list,n,y[i],y[j]);
    }
  }
  return aka;
};
