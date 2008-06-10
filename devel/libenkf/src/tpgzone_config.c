#include <stdbool.h>
#include <stdlib.h>
#include <config.h>
#include <ecl_grid.h>
#include <field.h>
#include <tpgzone_config.h>
#include <util.h>

/*
  IDEA FOR CONFIGURATION FILE:


  To add a truncated pluri-Gaussian zone, add a line

  TPGZONE ZONE_NAME CONFIG_FILE

  in the main configuration. ZONE_NAME will typically
  be something like "TABERT",  "LAYER2" etc.

  The zone configuration file CONFIG_FILE could look
  like:

  COORDS_BOX                    i1 i2 j1 j2 k1 k2
  FACIES_BACKGROUND             SHALE
  FACIES_FOREGROUND             SAND COAL CREVASSE
  NUM_GAUSS_FIELDS              2
  EROSION_SEQUENCE              EROSION_CONF_FILE
  PETROPHYSICS                  PETROPHYSICS_CONF_FILE 

  The idea is as follows:

  o First line spec's the zone. Here we can add other
    features later, such as COORDS_REGION and
    COORDS_LAYER. On allocation, we should warn about
    overlapping zones (but not abort, since it could
    be explicitly wanted!).

  o Second and third line specs the default facies
    and other facies. Alternatively, we could have
    this on one line and let the first facies be 
    the default. 

  o NUM_GAUSS_FIELDS specs the number of background
    Gaussian fields. Note that the number of fields
    can be different from the number of facies!
  

  o EROSION_SEQUENCE specs the erosion rule.
    This will be a separate file, which could look
    something like:
    
    SAND LINEAR a b c
    COAL LINEAR d e f

    This has the following interpreation:

    - The facies starts in the default facies, i.e.
      SHALE. If 
      
       a * g1 + b * g2 > c

      where g1 and g2 are the two Gaussian fields,
      then it is set to SAND. Furtermore, if

       d *g1 + e * g2 > f

      it's set to COAL.

    - I.e., the file specs and erosion sequence
      DOWNWARDS. Note that it could happen that
      one facies type is never used in this scheme.

  o PETEROPHYSICS specs the peterophysics rule.
    The file will typically look something like:

    SAND     PORO  UNIFORM    MIN MAX
    SAND     PERMX LOGNORMAL  MU SIGMA
    SAND     PERMZ LOGNORMAL  MU SIGMA
    SHALE    PORO  UNIFORM    MIN MAX
    SHALE    PERMX LOGNORMAL  MU SIGMA
    SHALE    PERMZ LOGNORMAL  MU SIGMA
    .....    ..... ........ ... ......
    .....    ..... ........ ... ......
    .....    ..... ........ ... ......
    .....    ..... ........ ... ......

  That's about it.. 
*/

/*
  This function creates a truncated pluri-Gaussian zone based
  on the six indicies of a box.
*/
tpgzone_config_type * tpgzone_config_alloc_from_box(const ecl_grid_type       * grid,
                                                    int                         num_gauss_fields,
                                                    int                         num_facies,
                                                    tpgzone_trunc_scheme_type * trunc_scheme,
                                                    const void_arg_type       * trunc_arg,
                                                    hash_type                 * facies_kw_hash,
                                                    scalar_config_type        * petrophysics,
                                                    int i1, int i2, int j1, int j2, int k1, int k2)
{
  /*
    Allocate the tpgzone_config_type struct.
  */
  tpgzone_config_type * config = util_malloc(sizeof * config,__func__);

  /*
    Set the obvious field of the tpgzone_config_type struct.
  */
  config->num_gauss_fields = num_gauss_fields;
  config->num_facies       = num_facies;
  config->trunc_scheme     = trunc_scheme;
  config->trunc_arg        = trunc_arg;
  config->facies_kw_hash   = facies_kw_hash;
  config->petrophysics     = petrophysics,
  config->write_compressed = true;

  /*
    From the macro CONFIG_STD_FIELDS
  */
  config->ecl_kw_name = NULL;

  /*
    OK, here comes the tricky part -  we need to calculate the
    linear index of the target nodes based on the indicies of the box.
    Note that we also account for inactive nodes.
  */
  {
    int nx,ny,nz,elements;
    ecl_box_type * box;

    /*
      Count the active elements in the tpgzone.
    */
    ecl_grid_get_dims(grid,&nx,&ny,&nz, NULL);
    box = ecl_box_alloc(nx,ny,nz,i1,i2,j1,j2,k1,k2);
    elements = ecl_grid_count_box_active(grid,box);
    config->num_active_blocks = elements;

    /*
      Set the active nodes
    */
    ecl_grid_set_box_active_list(grid,box,config->target_nodes);

    /*
      From the macro CONFIG_STD_FIELDS.
      
      Note that there are currently three extra parameters pr. facies,
      describing the poro, permx and permz.
    */
    config->data_size = 3 * num_facies + elements * num_gauss_fields;

    /*
      Free allocated memory.
    */
    ecl_box_free(box);
  }
  return config;
}




tpgzone_config_type * tpgzone_config_fscanf_alloc(const char * filename)
{
  tpgzone_config_type * tpgzone_config;

  config_type * config = config_alloc(false);

  /*
    Configure the config struct
  */
  config_init_item(config, "COORDS",            0, NULL, true, false, 0, NULL, 6, 6 , NULL);
  config_init_item(config, "FACIES_BACKGROUND", 0, NULL, true, false, 0, NULL, 1, 1 , NULL);
  config_init_item(config, "FACIES_FOREGROUND", 0, NULL, true, false, 0, NULL, 1, -1, NULL);
  config_init_item(config, "NUM_GAUSS_FIELDS",  0, NULL, true, false, 0, NULL, 1, 1 , NULL);
  config_init_item(config, "EROSION_SEQUENCE",  0, NULL, true, false, 0, NULL, 1, 1 , NULL);
  config_init_item(config, "PETROPHYSICS",      0, NULL, true, false, 0, NULL, 1, 1 , NULL);

  /*
    Parse the config
  */
  config_parse(config, filename, ENKF_COM_KW);

  
  config_free(config);
  return tpgzone_config;
}



/*
  Deallocator.
*/
void tpgzone_config_free(tpgzone_config_type * config)
{
  free(config->target_nodes);

  /*
    No reason to believe that this is set, but better safe than sorry.
  */
  if(config->ecl_kw_name != NULL)
  {
    free(config->ecl_kw_name);
  }
  free(config);

}



/*
  Set petrophysics.
*/

void tpgzone_config_set_petrophysics(const tpgzone_config_type * config,
                                      field_type                * target_poro,
                                      field_type                * target_permx,
                                      field_type                * target_permz)
{
  int index;
  const int num_active_blocks = config->num_active_blocks;  
  double * target_buffer = util_malloc(num_active_blocks * ecl_util_get_sizeof_ctype(ecl_double_type),__func__);

  for(index = 0; index < num_active_blocks; index++)
  {

  }

  free(target_buffer);
}
/*****************************************************************/

VOID_FREE(tpgzone_config)
GET_DATA_SIZE(tpgzone)
