#ifndef __TPGZONE_CONFIG_H__
#define __TPGZONE_CONFIG_H__
#include <stdbool.h>
#include <ecl_grid.h>
#include <enkf_macros.h>
#include <enkf_types.h>
#include <hash.h>
#include <tpgzone_trunc_scheme.h>
#include <void_arg.h>
#include <scalar_config.h>

typedef struct tpgzone_config_struct tpgzone_config_type;

/*****************************************************************/


struct tpgzone_config_struct
{
  CONFIG_STD_FIELDS;
  int                          num_gauss_fields;     /* Number of underlying Gaussian fields */
  int                          num_facies;           /* Number of facies */
  int                          num_active_blocks;    /* Number of active gridblocks in the zone */
  int                          num_target_fields;    /* Number of fields to be set based on the tpg */
  tpgzone_trunc_scheme_type  * trunc_scheme;         /* Pointer to the truncation scheme */
  hash_type                  * facies_kw_hash;       /* Pointer to a hash with facies keywords */
  scalar_config_type        ** petrophysics;         /* Petrophysics output transform for each target field and facies */
  int                        * target_nodes;         /* Linear index to grid nodes in the block */
  bool                         write_compressed;     /* Should stored output be compressed? */
};

void tpgzone_config_free(tpgzone_config_type *);

tpgzone_config_type * tpgzone_config_alloc_from_box(const ecl_grid_type        *,
                                                    int                         ,
                                                    int                         ,
                                                    int                         ,
                                                    tpgzone_trunc_scheme_type  *,
                                                    hash_type                  *,
                                                    scalar_config_type        **,
                                                    int, int, int, int, int, int);

tpgzone_config_type * tpgzone_config_fscanf_alloc(const char *, const ecl_grid_type *);

/*****************************************************************/

scalar_config_type ** tpgzone_config_petrophysics_fscanf_alloc(const char               *);
void                  tpgzone_config_petrophysics_write       (const tpgzone_config_type*);

/*****************************************************************/
VOID_FREE_HEADER(tpgzone_config);
GET_DATA_SIZE_HEADER(tpgzone);

#endif
