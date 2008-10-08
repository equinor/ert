#include <util.h>
#include <stdlib.h>
#include <string.h>
#include <lsf_request.h>


struct lsf_request_struct {
  char * request;
};


lsf_request_type * lsf_request_alloc() {
  lsf_request_type * lsf_request = util_malloc(sizeof * lsf_request , __func__);
  lsf_request->request           = NULL;

  return lsf_request;
}


void lsf_request_free(lsf_request_type * lsf_request) {
  lsf_request_reset(lsf_request);
  free( lsf_request );
}


void lsf_request_reset(lsf_request_type * lsf_request) {
  lsf_request->request = util_safe_free( lsf_request->request );
}


/**
   Current SUPER NAIVE implementation is simply based on strcat() of the various
   requests - good knows what the server will do with that?!
*/
void  lsf_request_update(lsf_request_type * lsf_request , const char * request) {
  if (request != NULL) {
    if (lsf_request->request == NULL)
      lsf_request->request = util_alloc_string_copy(request);
    else {
      int new_length = strlen(request) + strlen(lsf_request->request) + 2;
      lsf_request->request = util_realloc(lsf_request->request , new_length , __func__);
      strcat(lsf_request->request , " ");
      strcat(lsf_request->request , request);
    }
  }
}


const char * lsf_request_get(const lsf_request_type * lsf_request) {
  return lsf_request->request;
}
