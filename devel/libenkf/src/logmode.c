#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <logmode.h>
#include <enkf_types.h>
#include <util.h>

struct logmode_struct {
  double     base;
  bool       log_input;
  bool       log_enkf;
  bool       log_output;
};



static bool __iand(enkf_logmode_enum logmode , enkf_logmode_enum mask) {
  if (logmode & mask)
    return true;
  else
    return false;
}

  

logmode_type * logmode_alloc(double base , enkf_logmode_enum mode) {
  logmode_type * logmode = malloc(sizeof * logmode);
  logmode->base = base;
  logmode->log_input  = __iand(mode , log_input_mask);
  logmode->log_enkf   = __iand(mode , log_enkf_mask);
  logmode->log_output = __iand(mode , log_output_mask);
  return logmode;
}


void logmode_free(logmode_type * logmode) {
  free(logmode);
}


void logmode_fwrite(const logmode_type * logmode , FILE * stream) {
  enkf_logmode_enum mode = nolog;
  if (logmode->log_input)  mode += log_input_mask;
  if (logmode->log_enkf)   mode += log_enkf_mask;
  if (logmode->log_output) mode += log_output_mask;
  
  
  UTIL_FWRITE_SCALAR(logmode->base , stream);
  UTIL_FWRITE_SCALAR(mode          , stream);
}


logmode_type * logmode_fread_alloc(FILE * stream) {
  enkf_logmode_enum mode;
  double base;
  UTIL_FREAD_SCALAR(base , stream);
  UTIL_FREAD_SCALAR(mode , stream);
  return logmode_alloc(base , mode);
}



static void logmode_transform_data(double base, bool src_log , bool target_log , int size, double * data) {
  int i;
  double log_base = log(base);
  if (src_log) {
    if (!target_log)
      for (i=0; i < size; i++)
	data[i] = exp(log_base * data[i]);
  } else {
    if (target_log)
      for (i=0; i < size; i++)
	data[i] = log(data[i]) / log_base;
  }
}



void logmode_transform_input_data(const logmode_type * logmode , int size, double *data) {
  logmode_transform_data(logmode->base, logmode->log_input , logmode->log_enkf , size , data);
}


void logmode_transform_output_data(const logmode_type * logmode , int size, double *data) {
  logmode_transform_data(logmode->base , logmode->log_enkf , logmode->log_output , size , data);
}


double logmode_transform_output_scalar(const logmode_type * logmode , double data) {
  logmode_transform_data(logmode->base , logmode->log_enkf , logmode->log_output , 1 , &data);
  return data;
}


static void logmode_transform_distribution(double base , bool src_log , bool target_log , double *_mean , double *_std) {
  double mu,sigma;
  double mean = *_mean;
  double std  = *_std;
  
  if (src_log) {
    if (!target_log) {
      fprintf(stderr,"%s: transform not implemented - aborting \n",__func__);
      abort();

      /*
	Uncertain about the base ....
      */
      sigma = exp(log(base) * mean + std*std/2);
      std   = exp(log(base) * std*std + 2*mean)*(exp(log(base) * std*std) - 1);
    }
  } else if (target_log) {
    sigma = sqrt(log(std*std / (mean*mean) + 1));
    mu    = log(mean) - 0.5*log(std*std / (mean*mean) + 1);
    
    *_mean = mu    / log(base);
    *_std  = sigma / log(base);
  }
}



/*
  A variable x is lognormal distributed with: <x> = mean and <(x - <x>)^2> = std*std

  Then the variable y = log x is normally distributed with <y> = sigma_N and <(y - <y>)^2> = mu_N * mu_N
*/

void logmode_transform_input_distribution(const logmode_type * logmode , double *mean , double *std) {
  logmode_transform_distribution(logmode->base , logmode->log_input , logmode->log_enkf , mean , std);
}


bool logmode_logEnKF(const logmode_type * logmode) {  return logmode->log_enkf; }
