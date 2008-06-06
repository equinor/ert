#ifndef __TPGZONE_CONFIG_H__
#define __TPGZONE_CONFIG_H__
#include <stdbool.h>
#include <enkf_macros.h>
#include <enkf_types.h>
#include <hash.h>
#include <tpgzone_trunc_scheme.h>
#include <void_arg.h>
#include <scalar_config.h>

/*
  The heart of the truncated pluri-Gaussian model is the truncation
  scheme, which assigns a int based on some double variables.

  NOTE: 
  The truncation schemes are restricted to composition of some
  pre-defined parameteric functions. These *MUST* be hardcoded.
*/
//typedef int (tpgzone_trunc_scheme_type) (const double *, const void_arg_type *);

/*
  Shorthand notation for the tpgzone_config_struct.
*/
typedef struct tpgzone_config_struct tpgzone_config_type;



/*****************************************************************/



struct tpgzone_config_struct
{
  CONFIG_STD_FIELDS;
  int                         num_gauss_fields;     /* Number of underlying Gaussian fields */
  int                         num_facies;           /* Number of facies */
  int                         num_active_blocks;     /* Number of active gridblocks in the zone */
  tpgzone_trunc_scheme_type * trunc_scheme;         /* Pointer to the truncation scheme */
  const void_arg_type       * trunc_arg;            /* Pointer to the argument of the truncation scheme */
  hash_type                 * facies_kw_hash;       /* Pointer to a hash with facies keywords */
  scalar_config_type        * poro_trans;           /* Poro output transform for each facies */
  scalar_config_type        * permx_trans;          /* Permx output transform for each facies */
  scalar_config_type        * permz_trans;          /* Permx output transform for each facies */
  int                       * target_nodes;         /* Linear index to grid nodes in the block */
  bool                        write_compressed;     /* Should stored output be compressed? */
};

void tpgzone_config_type_free(tpgzone_config_type *);

/*****************************************************************/

VOID_FREE_HEADER(tpgzone_config);
GET_DATA_SIZE_HEADER(tpgzone);

#endif
