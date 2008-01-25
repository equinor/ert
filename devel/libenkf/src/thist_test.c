#include <thist.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>



double rand_normal(double mu , double std) {
  double R1 = rand() * 1.0 / RAND_MAX;
  double R2 = rand() * 1.0 / RAND_MAX;

  return mu + std * sqrt(-2.0 * log(R1)) * cos(2 * 3.14159265 * R2);
}


int main(int argc , char ** argv)  {
  const int  ens_size = 100;
  const int  length   = 25;
  int time_step , iens;
  
  double mu , std;
  double *forecast_data;
  double *analyzed_data;

  thist_type * thist = thist_alloc(length + 2, ens_size , thist_both , "Tittel");
  
  forecast_data = malloc(ens_size * sizeof * forecast_data);
  analyzed_data = malloc(ens_size * sizeof * analyzed_data);
  mu  = 0.0;
  std = 1.0;
  
  for (time_step = 0; time_step < length ; time_step++) {
    for (iens = 0; iens < ens_size; iens++) 
      forecast_data[iens] = rand_normal(mu , std);

    /*thist_update_vector_forecast(thist , time_step , forecast_data);*/

    mu  += rand_normal(0    , 0.15 * std);
    std *= rand_normal(0.75 , 0.10);
    for (iens = 0; iens < ens_size; iens++) 
      analyzed_data[iens] = rand_normal(mu , std);
    
    thist_update_vector_analyzed(thist , time_step , analyzed_data);
    
    mu  += rand_normal(0    , 0.50 * std);
    std *= rand_normal(1.10 , 0.15);
  }
  thist_matlab_dump(thist , "Testing/inspection/synthetic");
  printf("OK .... \n");
  thist_free(thist);
}


/*
  matlab code:

 function [s] = fread_string(fid)
   len = fread(fid , 1 , 'int')
   s   = fread(fid , len , 'uint8=>char')';

*/
