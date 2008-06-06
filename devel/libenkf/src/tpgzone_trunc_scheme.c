#include <tpgzone_trunc_scheme.h>

/*
  TODO

  Decide the structure for void_arg_type in case of a linear truncation scheme.

  Do we really need more than 5-10 underlying Gaussian fields?
*/
int trunc_scheme_func_linear(const double * gauss, int cur_facies, void_arg_type * arg)
{
  return 0;
}



int tpgzone_trunc_scheme_node_type_apply(tpgzone_trunc_scheme_node_type *node, int cur_facies, const double * gauss)
{
  return node->func(gauss,cur_facies,node->arg);
}



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
