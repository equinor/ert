/*
  This file implements a very small wrapper structure around the
  lsb_xxxx() functions from the libbat.so shared library which are
  used to submit, monitor and control simulations with LSF. 

  Loading and initializing the lsf libraries is quite painful, in an
  attempt to reduce unecessary dependencies the lsf libraries are
  loaded with dlopen() in the lsb_alloc() function below. This means
  that the libjob_queue.so shared library can be loaded without access
  to the lsf libraries.
*/

#include <stdlib.h>
#include <dlfcn.h>

//#include <lsf/lsbatch.h>

int main(int argc , char ** argv) {
  void * libnsl = NULL;//dlopen( "libnsl.so" , RTLD_NOW | RTLD_GLOBAL);
  void * liblsf = NULL;//dlopen( "liblsf.so" , RTLD_NOW | RTLD_GLOBAL);
   
   if ( libnsl && liblsf ) 
     exit(0);
   else
     exit(1); 
}
