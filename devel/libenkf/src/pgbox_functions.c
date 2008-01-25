#include <void_arg.h>


double pgfilter(const double *data , void_arg_type * arg) {
  double x = data[0];
  double y = data[1];
  
  return x + y;
}
