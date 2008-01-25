#include <math.h>
#include <trans_func.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <void_arg.h> 
#include <util.h>


/* This should be removed - old legacy shit from Oseberg East */
double trans_derrf_OE(double x , const void_arg_type * arg) {
  double y;
  int steps  = void_arg_get_int(arg , 0);
  double mu  = void_arg_get_double(arg , 1);
  double std = void_arg_get_double(arg , 2);
  y = floor(steps*0.5*(1 + erf(x/sqrt(2.0)))) / (steps - 1);
  return y;
}



double trans_errf(double x, const void_arg_type * arg) { 
  double min      = void_arg_get_double(arg , 0);
  double max      = void_arg_get_double(arg , 1);
  double skewness = void_arg_get_double(arg , 2);
  double width    = void_arg_get_double(arg , 3);
  
  double y = 0.5*(1 + erf((x + skewness)/(width * sqrt(2.0))));

  return min + y * (max - min);
}



double trans_const(double x , const void_arg_type * arg) { 
  return void_arg_get_double(arg , 0); 
}


/* Observe that the argument of the shift should be "+" */
double trans_derrf(double x , const void_arg_type * arg) {
  int    steps    = void_arg_get_int(arg , 0);
  double min      = void_arg_get_double(arg , 1);
  double max      = void_arg_get_double(arg , 2);
  double skewness = void_arg_get_double(arg , 3);
  double width    = void_arg_get_double(arg , 4);
  
  double y = floor( steps * 0.5*(1 + erf((x + skewness)/(width * sqrt(2.0)))) / (steps - 1) );
  return min + y * (max - min);
}



double trans_unif(double x , const void_arg_type * arg) {
  double y;
  double min   = void_arg_get_double(arg , 0);
  double max   = void_arg_get_double(arg , 1);
  y = 0.5*(1 + erf(x/sqrt(2.0))); /* 0 - 1 */
  return y * (max - min) + min;
}



double trans_dunif(double x , const void_arg_type * arg) {
  double y;
  int    steps = void_arg_get_int(arg , 0);
  double min   = void_arg_get_double(arg , 1);
  double max   = void_arg_get_double(arg , 2);
  
  y = 0.5*(1 + erf(x/sqrt(2.0))); /* 0 - 1 */
  return (floor( y * steps) / (steps - 1)) * (max - min) + min;
}



double trans_normal(double x , const void_arg_type * arg) {
  double mu , std;
  mu  = void_arg_get_double(arg , 0 );
  std = void_arg_get_double(arg , 1 );
  return x * std + mu;
}



transform_ftype * trans_func_lookup(FILE * stream , char ** _func_name , void_arg_type **_void_arg , bool *active) {
  char            * func_name;
  void_arg_type   * void_arg = NULL;
  transform_ftype * transf;

  *active = true;
  func_name = util_fscanf_alloc_token(stream);
  if (func_name == NULL) {
    fprintf(stderr,"%s: could not locate name of transformation - aborting \n",__func__);
    abort();
  }

  if (strcmp(func_name , "NORMAL") == 0) {
    /* Normal distribution */
    /* NORMAL mu std       */
    transf   = trans_normal;
    void_arg = void_arg_alloc2(double_value , double_value);
  } else if (strcmp(func_name , "UNIFORM") == 0) {
    /* Uniform distribution */
    /* UNIFORM min max      */
    transf   = trans_unif;
    void_arg = void_arg_alloc2(double_value , double_value);
  } else if (strcmp(func_name , "DUNIF") == 0) {
    /* DUNIF discrete uniform distribution */
    /* DUNIF steps min max */
    transf   = trans_dunif;
    void_arg = void_arg_alloc3(int_value , double_value , double_value);
  } else if (strcmp(func_name , "ERRF") == 0) {
    /* ERRF min max skewness width */
    transf   = trans_errf;
    void_arg = void_arg_alloc4(double_value , double_value , double_value , double_value);
  } else if (strcmp(func_name , "DERRF") == 0) {
    /* DERRF distribution */
    /* DUNIF steps min max skewness width */
    transf   = trans_derrf;
    void_arg = void_arg_alloc5(int_value , double_value , double_value , double_value , double_value);
  } else if (strcmp(func_name , "DERRF-OE") == 0) {
    /* DERRF-OE distribution - legacy shit from Oseberg East*/
    /* DUNIF steps mu std */
    transf   = trans_derrf_OE;
    void_arg = void_arg_alloc3(int_value , double_value , double_value);
  } else if (strcmp(func_name , "CONST") == 0) {
    /* Constant    */
    /* CONST value */
    transf   = trans_const;
    void_arg = void_arg_alloc1( double_value );
    *active  = false;
  } else {
    fprintf(stderr,"%s: function name:%s not recognized - aborting \n", __func__ , func_name);
    abort();
  }
  void_arg_fscanf(void_arg , stream);
  
  *_func_name = func_name;
  *_void_arg  = void_arg;
  return transf;
}







