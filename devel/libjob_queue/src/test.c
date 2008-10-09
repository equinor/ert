#include <stdlib.h>
#include <lsf_request.h>




int main(int argc , char ** argv) {
  
  lsf_request_type * lsf_request = lsf_request_alloc(NULL , NULL);
  
  lsf_request_update__(lsf_request , "rusage[eclipse=1:duration=10] select[ia64||ia32]");
  lsf_request_update__(lsf_request , "rusage[wsegs=1] select[ia64||x86_64]");
  lsf_request_update__(lsf_request , "rusage[eclipse=1:duration=10] select[ia64]");
  lsf_request_set_request_string(lsf_request);

  printf("Resulting request: %s \n",lsf_request_get(lsf_request));
  
  
}
