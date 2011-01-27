/**
   If the symbol INCLUDE_LSF is not defined a dummy LSF driver will be
   compiled. This driver has stubs for all the driver related
   functions, and a dummy driver can be created with the
   queue_driver_alloc_LSF() function.

   If one of the queue_driver function poiniters pointing down to one
   of the lsf_driver_xxx() functions is actually invokes (e.g. through
   the queue layer) proper driver functions are used the program will
   exit with an error message. This is only a utility to avoid
   changing the source when the library is built and used on a
   platform without LSF installed.

   When compiling with LSF support the preprocessor symbol INCLUDE_LSF
   must be set (to an arbitrary value), in addition the libraries
   liblsf, libbat and libnsl must be linked in when creating the final
   executable.  
*/

#ifdef INCLUDE_LSF
#include "lsf_driver_impl.c"
#else
#include "lsf_driver_dummy.c"
#endif
