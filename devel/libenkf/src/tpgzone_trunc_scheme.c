#include <tpgzone_trunc_scheme.h>



struct tpgzone_trunc_scheme_node_type_struct
{
  tpgzone_trunc_scheme_func_type * func;
  void_arg_type                  * arg;
};



struct tpgzone_trunc_scheme_type_struct
{
  int                               num_nodes;
  tpgzone_trunc_scheme_node_type ** trunc_scheme_nodes;
};



/*****************************************************************/



/*
  TODO

  Decide the structure for void_arg_type in case of a linear truncation scheme.

  Do we really need more than 5-10 underlying Gaussian fields? Or can be use the 
  void_arg_type with some double values? The first field in arg should at least
  be the number of Gaussian fields.
*/
int trunc_scheme_func_linear(const double * gauss, int cur_facies, void_arg_type * arg)
{
  /*
    TODO

    Make this do something..
  */
  return cur_facies;
}



/*
 Think of this as an "erosion".

 Given value of the Gaussian fields stored in the double pointer,
 and the current facies type stored in int, a new facies type is returned
 according to an erosion rule with parameters stored in void_arg_type.
*/
int tpgzone_trunc_scheme_node_type_apply(tpgzone_trunc_scheme_node_type *node, int cur_facies, const double * gauss)
{
  return node->func(gauss,cur_facies,node->arg);
}



/*
  .. and of this as an erosion sequence.
*/
int tpgzone_trunc_scheme_type_apply(tpgzone_trunc_scheme_type *trunc_scheme, const double *gauss)
{
  int i, facies;

  /*
    Initialize to default facies (i.e. 0)
  */
  facies = 0;

  /*
    Apply the "erosion sequence"
  */
  for(i=0; i<trunc_scheme->num_nodes; i++)
  {
    facies = tpgzone_trunc_scheme_node_type_apply(trunc_scheme->trunc_scheme_nodes[i], facies, gauss);
  }

  return facies;
}
