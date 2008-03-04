#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <util.h>
#include "cost_func.h"
#include "pre-image.h"
#include "gnuplot.h"


extern void m_sample2d_mp_sample2d_(double*,        //A
                                    const int*,     //nx
                                    const int*,     //ny
                                    const int*,     //nrens
                                    const int*,     //nre
                                    const double*,  //dx
                                    const double*,  //dy
                                    const double*,  //rx
                                    const double*,  //ry
                                    const double*,  //dir
                                    const int*,     //sampfix
                                    const int*      //verbose
                                    );

extern void m_pseudo2d_mp_pseudo2d_(double *,       //A
                                    const int *,    //nx
                                    const int *,    //ny
                                    const int *,    //lde
                                    const double *, //rx
                                    const double *, //ry
                                    const double *, //dx
                                    const double *, //dy
                                    const int *,    //n1
                                    const int *,    //n2
                                    const double *, //theta
                                    const int *     //verbose
                                    );

void box_muller(const int n, double *s, const double var, const double mean)
{
  int i;
  double u1,u2;
  for(i=0; i<n;i++)
  {
    u1 = (double) (rand() + 1 )/ (double) RAND_MAX;
    u2 = (double) (rand() + 1 )/ (double) RAND_MAX;
    s[i] = sqrt(var) * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2) + mean;
  }
};



double min(const double x, const double y)
{
  if(x < y)
  {
    return x;
  }
  else
  {
    return y;
  }
};



double max(const double x, const double y)
{
  if(x > y)
  {
    return x;
  }
  else
  {
    return y;
  }
};



int main(int argc,char **argv)
{
  const int nx = 400;
  const int ny = 400;
  const int n = nx*ny;
  const int nre = 1;
  const int ns = 100;

  const double dx = 25.0;
  const double dy = 25.0;
  const double rx = 300.0;
  const double ry = 100;
  const double dir = 40.0;
  const int verbose = -1;
  const int sampfix = -1;

  int i;
  int ntries = 15;

  double lamda;

  double *x;
  double *mu;

  int *nbd;
  double *low_bnd;
  double *high_bnd;
  double *alpha;
  double **y;


  // Set seed
  if(argc > 1)
  {
    srand(atoi(argv[1]));
  }


  // Unbuffered output
  setbuf(stdout,NULL);

  //Configure cost function and kernels
  f_func f = {XPLUSSQRTX};
  reg_func reg = {NONE};

  // 2nd order dot product
  cost_func_type *cost_func_2xdot;
  kernel_list_type *kernel_2xdot;
  kernel_type names2xdot[2] = {DOT,DOT};
  double weights2xdot[2] = {1.0,1.0};
  double params2xdot[2] = {1.0,2.0};
  kernel_2xdot = kernel_list_alloc(2,names2xdot,weights2xdot,params2xdot);
  kernel_list_assert(kernel_2xdot);
  cost_func_2xdot = cost_func_alloc(kernel_2xdot,f,reg);

  // 3rd order dot product
  cost_func_type *cost_func_3xdot;
  kernel_list_type *kernel_3xdot;
  kernel_type names3xdot[3] = {DOT,DOT,DOT};
  double weights3xdot[3] = {1.0,1.0,1.0};
  double params3xdot[3] = {1.0,2.0,3.0};
  kernel_3xdot = kernel_list_alloc(3,names3xdot,weights3xdot,params3xdot);
  kernel_list_assert(kernel_3xdot);
  cost_func_3xdot = cost_func_alloc(kernel_3xdot,f,reg);

  // Gauss
  cost_func_type *cost_func_gauss;
  kernel_list_type *kernel_gauss;
  kernel_type namesgauss[1] = {GAUSS};
  double weightsgauss[1] = {1.0};
  double paramsgauss[1] = {(double) n};
  kernel_gauss = kernel_list_alloc(1,namesgauss,weightsgauss,paramsgauss);
  kernel_list_assert(kernel_gauss);
  cost_func_gauss = cost_func_alloc(kernel_gauss,f,reg);



  //Allocate the rest of the workspace
  x         = util_malloc(n     *sizeof *x,       __func__);
  mu        = util_malloc(n     *sizeof *mu,      __func__);
  low_bnd   = util_malloc(n     *sizeof *low_bnd, __func__);
  high_bnd  = util_malloc(n     *sizeof *high_bnd,__func__);
  nbd       = util_malloc(n     *sizeof *nbd,     __func__);
  alpha     = util_malloc(ns    *sizeof *alpha,   __func__);

  //Initialize

  for(i=0; i<n; i++)
  {
    x[i]        = 0.0;
    mu[i]       = 0.0;
    low_bnd[i]  = 0.0;
    high_bnd[i] = 0.0;
    nbd[i]      = 0;
  }

  for(i=0; i<ns; i++)
  {
    alpha[i] = 0.0;
  }
  
  //We are going to use fortran to fill y, therefore it must
  //be one continious block of memory
  y         = util_malloc(ns    *sizeof *y,       __func__);
  y[0]      = util_malloc(ns*n  *sizeof **y,      __func__);

  //Fix the indicies in y
   for(i=1; i<ns;i++)
  {
    y[i] = y[i-1] + n;
  }

  //Sample y and alpha
  box_muller(ns,alpha,10 / (double) ns,0.0);
  alpha[0] = alpha[0] + 1;
  m_sample2d_mp_sample2d_(y[0],&nx,&ny,&ns,&nre,&dx,&dy,&rx,&ry,&dir,&sampfix,&verbose);

  // Write some samples to file
  gnuplot_write_ungridded(nx,ny,y[0],"unbounded_sample_0.plt");
  gnuplot_write_ungridded(nx,ny,y[1],"unbounded_sample_1.plt");

  // Find an aproximative pre-image using the 2xdot cost function
  pre_image_approx(cost_func_2xdot, ntries, n,ns,(const double**) y,alpha,lamda,mu, low_bnd,high_bnd,nbd,x);
  gnuplot_write_ungridded(nx,ny,x,"unbounded_pre-image_2xdot.plt");

  // Find an aproximative pre-image using the 3xdot cost function
  pre_image_approx(cost_func_3xdot, ntries, n,ns,(const double**) y,alpha,lamda,mu, low_bnd,high_bnd,nbd,x);
  gnuplot_write_ungridded(nx,ny,x,"unbounded_pre-image_3xdot.plt");

  // Find an aproximative pre-image using the gauss cost function
  pre_image_approx(cost_func_gauss, ntries, n,ns,(const double**) y,alpha,lamda,mu, low_bnd,high_bnd,nbd,x);
  gnuplot_write_ungridded(nx,ny,x,"unbounded_pre-image_gauss.plt");

  // Truncate y to [0,1]
  for(i=0; i<ns*n; i++)
  {
    y[0][i] = min(y[0][i],1.0);
    y[0][i] = max(y[0][i],0.0);
 //   y[0][i] = (double) ceil(y[0][i]);
  }

  // Set the bounds
  for(i=0; i<n; i++)
  {
    low_bnd[i]  = 0.0;
    high_bnd[i] = 1.0;
    nbd[i]      = 2;
  }

  // Write two of the bounded samples to file
  gnuplot_write_ungridded(nx,ny,y[0],"bounded_sample_0.plt");
  gnuplot_write_ungridded(nx,ny,y[1],"bounded_sample_1.plt");

  // Find an aproximative pre-image using the 2xdot cost-function
  pre_image_approx(cost_func_2xdot, ntries, n,ns,(const double**) y,alpha,lamda,mu, low_bnd,high_bnd,nbd,x);
  gnuplot_write_ungridded(nx,ny,x,"bounded_pre-image_2xdot.plt");

  // Find an aproximative pre-image using the 3xdot cost-function
  pre_image_approx(cost_func_3xdot, ntries, n,ns,(const double**) y,alpha,lamda,mu, low_bnd,high_bnd,nbd,x);
  gnuplot_write_ungridded(nx,ny,x,"bounded_pre-image_3xdot.plt");

  // Find an aproximative pre-image using the gauss cost-function
  pre_image_approx(cost_func_gauss, ntries, n,ns,(const double**) y,alpha,lamda,mu, low_bnd,high_bnd,nbd,x);
  gnuplot_write_ungridded(nx,ny,x,"bounded_pre-image_gauss.plt");
  
  // Free allocated memory
  free(x);
  free(mu);
  free(low_bnd);
  free(high_bnd);
  free(nbd);
  free(alpha);
  kernel_list_free(kernel_2xdot);
  cost_func_free(cost_func_2xdot);
  kernel_list_free(kernel_3xdot);
  cost_func_free(cost_func_3xdot);
  kernel_list_free(kernel_gauss);
  cost_func_free(cost_func_gauss);
  free(y[0]);
  free(y);
  return 0;
};
