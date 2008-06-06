#ifndef __TPGZONE_TRUNC_SCHEME_H__
#define __TPGZONE_TRUNC_SCHEME_H__

#include <void_arg.h>

/*
 Think of this as an erosion.

 Given value of the Gaussian fields stored in the double pointer,
 and the current facies type stored in int, a new facies type is returned
 according to an erosion rule with parameters stored in void_arg_type.
*/
typedef int (tpgzone_trunc_scheme_func_type)(const double *,int, void_arg_type *);

typedef struct tpgzone_trunc_scheme_node_struct tpgzone_trunc_scheme_node_type;
typedef struct tpgzone_trunc_scheme_type_struct tpgzone_trunc_scheme_type;

/*
  TODO

  Decide if these structs can be private for type safety.
*/
struct tpgzone_trunc_scheme_node_struct
{
  tpgzone_trunc_scheme_func_type * func;
  void_arg_type                  * arg;
};

struct tpgzone_trunc_scheme_type_struct
{
  int                               num_nodes;
  tpgzone_trunc_scheme_node_type ** trunc_scheme_nodes;
};

#endif
