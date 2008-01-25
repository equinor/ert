#include <stdlib.h>
#include <stdio.h>
#include <enkf_util.h>
#include <obs_data.h>
#include <util.h>
#include <analysis.h>

struct obs_data_struct {
  int      size;
  int      alloc_size;
  double  *value;
  double  *std;
  char   **keyword;
}; 


static void obs_data_realloc_data(obs_data_type * obs_data, int new_alloc_size) {
  int old_alloc_size = obs_data->alloc_size;
  obs_data->alloc_size = new_alloc_size;
  obs_data->value      = enkf_util_realloc(obs_data->value   , new_alloc_size * sizeof * obs_data->value   , __func__);
  obs_data->std        = enkf_util_realloc(obs_data->std     , new_alloc_size * sizeof * obs_data->std     , __func__);
  obs_data->keyword    = enkf_util_realloc(obs_data->keyword , new_alloc_size * sizeof * obs_data->keyword , __func__);
  {
    int i;
    for (i= old_alloc_size; i < new_alloc_size; i++)
      obs_data->keyword[i] = NULL;
  }
}



obs_data_type * obs_data_alloc() {
  obs_data_type * obs_data = malloc(sizeof * obs_data);
  obs_data->size       = 0;
  obs_data->value      = NULL;
  obs_data->std        = NULL;
  obs_data->keyword    = NULL;

  obs_data->alloc_size = 0;
  obs_data_realloc_data(obs_data , 10);
  return obs_data;
}



void obs_data_reset(obs_data_type * obs_data) { obs_data->size = 0; }


void obs_data_add(obs_data_type * obs_data, double value, double std , const char *kw) {
  if (obs_data->size == obs_data->alloc_size)
    obs_data_realloc_data(obs_data , 2*obs_data->alloc_size + 2);
  {
    int index = obs_data->size;
    obs_data->value[index]   = value;
    obs_data->std[index]     = std;
    obs_data->keyword[index] = util_realloc_string_copy(obs_data->keyword[index] , kw);
  }
  obs_data->size++;
}



void obs_data_free(obs_data_type * obs_data) {
  free(obs_data->value);
  free(obs_data->std);
  util_free_string_list(obs_data->keyword , obs_data->size);
  free(obs_data);
}



void obs_data_fprintf(const obs_data_type * obs_data , FILE *stream) {
  int i;
  for (i = 0; i < obs_data->size; i++)
    fprintf(stream , "%-3d : %-16s  %12.3f +/- %12.3f \n", i+1 , obs_data->keyword[i] , obs_data->value[i] , obs_data->std[i]);
}



double * obs_data_allocD(const obs_data_type * obs_data , int ens_size, const double * S , const double * meanS) {
  double *D;
  int iens, iobs, nrobs;
  int ens_stride , obs_stride;
  nrobs = obs_data->size;
  analysis_set_stride(ens_size , nrobs , &ens_stride , &obs_stride);
  D = util_malloc(nrobs * ens_size * sizeof * D , __func__);
  for  (iens = 0; iens < ens_size; iens++) {
    for (iobs = 0; iobs < nrobs; iobs++) {
      int index = iens * ens_stride + iobs * obs_stride;
      D[index] = obs_data->value[iobs] - (S[index] + meanS[iobs]);
    }
  }
  return D;
}


