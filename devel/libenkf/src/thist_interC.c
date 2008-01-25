#include <stdlib.h>
#include <thist.h>
#include <util.h>

static thist_type * GLOBAL_THIST = NULL;

void thist_init__(const int * ens_size , const int * data_type) {
  GLOBAL_THIST = thist_alloc(100 , *ens_size , *data_type , NULL);
}


void thist_update_scalar__(const int * time_step , const int * iens , const double * forecast , const double * analyzed) {
  thist_update_scalar(GLOBAL_THIST , *time_step , *iens , *forecast , *analyzed);
}

void thist_update_scalar_forecast__(const int * time_step , const int * iens , const double * forecast) {
  thist_update_scalar_forecast(GLOBAL_THIST , *time_step , *iens , *forecast);
}

void thist_update_scalar_analyzed__(const int * time_step , const int * iens , const double * analyzed ) {
  thist_update_scalar_analyzed(GLOBAL_THIST , *time_step , *iens , *analyzed);
}


void thist_update_vector__(const int * time_step , const double * forecast , const double * analyzed) {
  thist_update_vector(GLOBAL_THIST , *time_step , forecast , analyzed);
}

void thist_update_vector_forecast__(const int * time_step , const double * forecast) {
  thist_update_vector_forecast(GLOBAL_THIST , *time_step , forecast);
}

void thist_update_vector_analyzed__(const int * time_step , const double * analyzed) {
  thist_update_vector_analyzed(GLOBAL_THIST , *time_step , analyzed);
}


void thist_matlab_dump__(const char * _file , const int * file_len , const char * _title , const int * title_len) {
  char * file  = util_alloc_cstring(_file  , file_len);
  char * title = util_alloc_cstring(_title , title_len);

  printf("Skal sette title: %s \n",title);
  thist_set_title(GLOBAL_THIST , title);
  thist_matlab_dump(GLOBAL_THIST , file);

  free(file);
  free(title);
}


void thist_clear__() {
  thist_clear(GLOBAL_THIST);
}



void thist_free__() {
  thist_free(GLOBAL_THIST);
}
