#include <math.h>
#include <trans_func.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <arg_pack.h> 
#include <util.h>





/**
   Width  = 1 => uniform
   Width  > 1 => unimodal peaked
   Width  < 1 => bimoal peaks


   Skewness < 0 => shifts towards the left
   Skewness = 0 => symmetric
   Skewness > 0 => Shifts towards the right

   The width is a relavant scale for the value of skewness.
*/

double trans_errf(double x, const arg_pack_type * arg) { 
  double min      = arg_pack_iget_double(arg , 0);
  double max      = arg_pack_iget_double(arg , 1);
  double skewness = arg_pack_iget_double(arg , 2);
  double width    = arg_pack_iget_double(arg , 3);
  double y;

  y = 0.5*(1 + erf((x + skewness)/(width * sqrt(2.0))));
  return min + y * (max - min);
}


void trans_errf_check(const char * func_name , const arg_pack_type * arg) {
  double width    = arg_pack_iget_double(arg , 3);
  if (width <= 0)
    util_exit("In function:%s the witdh must be > 0.",func_name);
}





double trans_const(double x , const arg_pack_type * arg) { 
  return arg_pack_iget_double(arg , 0); 
}


/* Observe that the argument of the shift should be "+" */
double trans_derrf(double x , const arg_pack_type * arg) {
  int    steps    = arg_pack_iget_int(arg , 0);
  double min      = arg_pack_iget_double(arg , 1);
  double max      = arg_pack_iget_double(arg , 2);
  double skewness = arg_pack_iget_double(arg , 3);
  double width    = arg_pack_iget_double(arg , 4);
  double y;
  
  y = floor( steps * 0.5*(1 + erf((x + skewness)/(width * sqrt(2.0)))) / (steps - 1) );
  return min + y * (max - min);
}


void trans_derrf_check(const char * func_name , const arg_pack_type * arg) {
  int    steps    = arg_pack_iget_int(arg , 0);
  double width    = arg_pack_iget_double(arg , 4);
  if (width <= 0)
    util_exit("In function:%s the witdh must be > 0.",func_name);

  if (steps <= 1)
    util_exit("In function:%s the number of steps must be greater than 1.",func_name);
}





double trans_unif(double x , const arg_pack_type * arg) {
  double y;
  double min   = arg_pack_iget_double(arg , 0);
  double max   = arg_pack_iget_double(arg , 1);
  y = 0.5*(1 + erf(x/sqrt(2.0))); /* 0 - 1 */
  return y * (max - min) + min;
}



double trans_dunif(double x , const arg_pack_type * arg) {
  double y;
  int    steps = arg_pack_iget_int(arg , 0);
  double min   = arg_pack_iget_double(arg , 1);
  double max   = arg_pack_iget_double(arg , 2);
  
  y = 0.5*(1 + erf(x/sqrt(2.0))); /* 0 - 1 */
  return (floor( y * steps) / (steps - 1)) * (max - min) + min;
}


void trans_dunif_check(const char * func_name , const arg_pack_type * arg) {
  int    steps = arg_pack_iget_int(arg , 0);
  
  if (steps <= 1)
    util_exit("When using function:%s steps must be > 1 \n",func_name);
}



double trans_normal(double x , const arg_pack_type * arg) {
  double mu , std;
  mu  = arg_pack_iget_double(arg , 0 );
  std = arg_pack_iget_double(arg , 1 );
  return x * std + mu;
}



double trans_lognormal(double x, const arg_pack_type * arg) {
  double mu, std;
  mu  = arg_pack_iget_double(arg , 0 );
  std = arg_pack_iget_double(arg , 1 );
  return exp(x * std + mu);
}



/**
   Used to sample values between min and max - BUT it is the logarithm
   of y which is uniformly distributed. Relates to the uniform
   distribution in the same manner as the lognormal distribution
   relates to the normal distribution.
*/
double trans_logunif(double x , const arg_pack_type * arg) {
  double log_min = log(arg_pack_iget_double(arg , 0));
  double log_max = log(arg_pack_iget_double(arg , 1));
  double log_y;
  {
    double tmp = 0.5*(1 + erf(x/sqrt(2.0)));           /* 0 - 1 */
    log_y      = log_min + tmp * (log_max - log_min);  /* Shift according to max / min */
  } 
  return exp(log_y);
}


void trans_logunif_check(const char * func_name , const arg_pack_type * arg) {
  double min = arg_pack_iget_double(arg , 0);
  double max = arg_pack_iget_double(arg , 1);
  if (min <= 0 || max <= 0)
    util_exit("When using:%s both arguments must be greater than zero.\n",func_name);
}



transform_ftype * trans_func_lookup(FILE * stream , char ** _func_name , arg_pack_type **_arg_pack) {
  char            * func_name;
  arg_pack_type   * arg_pack = NULL;
  transform_ftype * transf   = NULL;

  func_name = util_fscanf_alloc_token(stream);
  if (func_name == NULL) 
    util_abort("%s: could not locate name of transformation - aborting \n",__func__);
  
  arg_pack = arg_pack_alloc();
  if (strcmp(func_name , "NORMAL") == 0) {
    /* Normal distribution */
    /* NORMAL mu std       */
    transf   = trans_normal;
    arg_pack_append_double(arg_pack , 0);
    arg_pack_append_double(arg_pack , 0);
  } else if (strcmp(func_name , "LOGNORMAL") == 0) {
    /* Log normal distribution */
    /* LOGNORMAL mu std      */
    transf   = trans_lognormal;
    arg_pack_append_double(arg_pack , 0);
    arg_pack_append_double(arg_pack , 0);
  } else if (strcmp(func_name , "UNIFORM") == 0) {
    /* Uniform distribution */
    /* UNIFORM min max      */
    transf   = trans_unif;
    arg_pack_append_double(arg_pack , 0);
    arg_pack_append_double(arg_pack , 0);
  } else if (strcmp(func_name , "DUNIF") == 0) {
    /* DUNIF discrete uniform distribution */
    /* DUNIF steps min max */
    transf   = trans_dunif;
    arg_pack_append_int(arg_pack , 0);
    arg_pack_append_double(arg_pack , 0);
    arg_pack_append_double(arg_pack , 0);
    trans_dunif_check("DUNIF" , arg_pack);
  } else if (strcmp(func_name , "ERRF") == 0) {
    /* ERRF min max skewness width */
    transf   = trans_errf;
    arg_pack_append_double(arg_pack , 0);
    arg_pack_append_double(arg_pack , 0);
    arg_pack_append_double(arg_pack , 0);
    arg_pack_append_double(arg_pack , 0);
    trans_errf_check("ERRF" , arg_pack);
  } else if (strcmp(func_name , "DERRF") == 0) {
    /* DERRF distribution */
    /* DUNIF steps min max skewness width */
    transf   = trans_derrf;
    arg_pack_append_double(arg_pack , 0);
    arg_pack_append_double(arg_pack , 0);
    arg_pack_append_double(arg_pack , 0);
    arg_pack_append_double(arg_pack , 0);
    arg_pack_append_double(arg_pack , 0);
  } else if (strcmp(func_name , "LOGUNIF") == 0) {
    /* ULOG min max */
    transf   = trans_logunif;
    arg_pack_append_double(arg_pack , 0);
    arg_pack_append_double(arg_pack , 0);
    trans_logunif_check("LOGUNIF" , arg_pack);
  } else if (strcmp(func_name , "CONST") == 0) {
    /* Constant    */
    /* CONST value */
    transf   = trans_const;
    arg_pack_append_double(arg_pack , 0);
  } else 
    util_abort("%s: function name:%s not recognized - aborting \n", __func__ , func_name);
  
  arg_pack_fscanf(arg_pack , stream);
  arg_pack_lock( arg_pack );
  
  *_func_name = func_name;
  *_arg_pack  = arg_pack;
  return transf;
}







