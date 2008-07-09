#include <util.h>
#include <tpgzone_trunc_scheme.h>



struct tpgzone_trunc_scheme_node_type_struct
{
  tpgzone_trunc_scheme_func_type * func;       /* Truncation function */
  double                         * arg;        /* Arguments to func   */
  int                              new_facies; /* New facies type on func true */
};



struct tpgzone_trunc_scheme_type_struct
{
  int                               num_nodes;
  int                               num_gauss_fields;
  tpgzone_trunc_scheme_node_type ** trunc_scheme_nodes;
};



/*****************************************************************/



bool trunc_scheme_func_linear(const double * gauss, double * arg, int num_gauss_fields)
{
  int i;
  double val;

  val = 0.0;
  for(i=0; i<num_gauss_fields; i++)
    val = val + gauss[i] * arg[i];

  if(val >= arg[num_gauss_fields])
    return true;
  else
    return false;
}



/*
 Think of this as an "erosion".

 Given value of the Gaussian fields stored in the double pointer,
 and the current facies type stored in cur_facies, a new facies type is returned
 according to an erosion rule stored in node.
*/
int tpgzone_trunc_scheme_node_type_apply(tpgzone_trunc_scheme_node_type *node,
                                        int cur_facies, const double * gauss, int num_gauss_fields)
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



tpgzone_trunc_scheme_type * tpgzone_trunc_scheme_type_fscanf_alloc(const char       * filename,
                                                                   const hash_type  * facies_kw_hash,
                                                                   int                num_gauss_fields)
{
  FILE * stream;

  stream = util_fopen(filename,"r");


  fclose(stream); 
  return NULL;
};



tpgzone_trunc_scheme_node_type * tpgzone_trunc_scheme_type_fscanf_alloc_node(FILE            * stream,
                                                                             const hash_type * facies_kw_hash,
                                                                             int               num_gauss_fields)
{
  return NULL;
}
