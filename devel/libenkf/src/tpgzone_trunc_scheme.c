#include <tpgzone_trunc_scheme.h>



struct tpgzone_trunc_scheme_node_type_struct
{
  tpgzone_trunc_scheme_func_type * func;       /* Truncation function */
  double                         * arg;        /* Arguments to func   */
  int                              new_facies; /* If func is evaluated to be true, this is the new facies type */
};



struct tpgzone_trunc_scheme_type_struct
{
  int                               num_nodes;
  int                               num_gauss_fields;
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
bool trunc_scheme_func_linear(const double * gauss, double * arg, int num_gauss_fields)
{
  printf("%s: *WARNING* this function is empty.\n",__func__);
  return false;
}



/*
 Think of this as an "erosion".

 Given value of the Gaussian fields stored in the double pointer,
 and the current facies type stored in int, a new facies type is returned
 according to an erosion rule with parameters stored in void_arg_type.
*/
int tpgzone_trunc_scheme_node_type_apply(tpgzone_trunc_scheme_node_type *node, int cur_facies, const double * gauss, int num_gauss_fields)
{
  if(node->func(gauss, node->arg, num_gauss_fields))
    return node->new_facies;
  else
    return cur_facies;
}



/*
  .. and of this as an erosion sequence.
*/
int tpgzone_trunc_scheme_type_apply(tpgzone_trunc_scheme_type *trunc_scheme, const double *gauss)
{
  int i, facies;

  facies = 0;

  for(i=0; i<trunc_scheme->num_nodes; i++)
  {
    facies = tpgzone_trunc_scheme_node_type_apply(trunc_scheme->trunc_scheme_nodes[i], facies, gauss, trunc_scheme->num_gauss_fields);
  }

  return facies;
}



tpgzone_trunc_scheme_type * tpgzone_trunc_scheme_type_fscanf_alloc(const char * filename,
                                                                   hash_type  * facies_kw_hash,
                                                                   int         num_gauss_fields)
{
  printf("*WARNING*: %s is **NOT** fully implemented.\n",__func__);
  printf("Getting facies_kw_hash with keywords:\n");
  hash_printf_keys(facies_kw_hash);

  printf("\nI am asked to load a truncation scheme from %s\n\n",filename);
  return NULL;
};
