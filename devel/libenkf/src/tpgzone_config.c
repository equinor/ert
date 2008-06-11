#include <stdbool.h>
#include <stdlib.h>
#include <config.h>
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
  TRUNCATION_SEQUENCE           TRUNCATION_CONF_FILE
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
  

  o TRUNCATION_SEQUENCE specs the truncation rule.
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

    - I.e., the file specs and truncation sequence
      DOWNWARDS. Note that it could happen that
      one facies type is never used in this scheme.

  o PETEROPHYSICS specs the peterophysics rule.
    The file will typically look something like:

    SAND     TARGET_FIELD_01  UNIFORM    MIN MAX
    SAND     TARGET_FIELD_02  LOGNORMAL  MU SIGMA
    SAND     TARGET_FIELD_03  LOGNORMAL  MU SIGMA
    SHALE    TARGET_FIELD_01  UNIFORM    MIN MAX
    SHALE    TARGET_FIELD_02  LOGNORMAL  MU SIGMA
    SHALE    TARGET_FIELD_03  LOGNORMAL  MU SIGMA
    .....    ..... ........ ... ......
    .....    ..... ........ ... ......
    .....    ..... ........ ... ......
    .....    ..... ........ ... ......

    Here, TARGET_FIELD_* are arbitrary fields. E.g.,
    MY_PORO or MY_PERMX_MULTIPLIER.
    
    NOTE: If  is used with one facies type,

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
                                                    hash_type                 * facies_kw_hash,
                                                    scalar_config_type        * petrophysics,
                                                    int i1, int i2, int j1, int j2, int k1, int k2)
{
  tpgzone_config_type * config = util_malloc(sizeof * config,__func__);

  config->num_gauss_fields = num_gauss_fields;
  config->num_facies       = num_facies;
  config->trunc_scheme     = trunc_scheme;
  config->facies_kw_hash   = facies_kw_hash;
  config->petrophysics     = petrophysics,
  config->write_compressed = true;
  config->ecl_kw_name      = NULL;

  /*
    OK, here comes the tricky part -  we need to calculate the
    linear index of the target nodes based on the indicies of the box.
    Note that we also account for inactive nodes.
  */
  {
    int nx,ny,nz,elements;
    ecl_box_type * box;

    ecl_grid_get_dims(grid,&nx,&ny,&nz, NULL);

    box                       = ecl_box_alloc(nx,ny,nz,i1,i2,j1,j2,k1,k2);
    elements                  = ecl_grid_count_box_active(grid,box);
    config->num_active_blocks = elements;

    ecl_grid_set_box_active_list(grid,box,config->target_nodes);

    /*
      From the macro CONFIG_STD_FIELDS.
      
      Note that there are currently three extra parameters pr. facies,
      describing the poro, permx and permz.
    */
    config->data_size = 3 * num_facies + elements * num_gauss_fields;

    ecl_box_free(box);
  }
  return config;
}




/*
  Allocate a truncated pluri-Gaussian zone from a file.
*/
tpgzone_config_type * tpgzone_config_fscanf_alloc(const char * filename)
{
  tpgzone_config_type       * tpgzone_config;
  hash_type                 * facies_kw_hash  = hash_alloc();
  tpgzone_trunc_scheme_type * trunc_scheme;
  scalar_config_type        * petrophysics;
  config_type               * config          = config_alloc(false);

  char * truncation_sequence_conf_file;
  char * peterophysics_conf_file;

  config_init_item(config, "COORDS",               0, NULL, true, false, 0, NULL, 6, 6 , NULL);
  config_init_item(config, "FACIES_BACKGROUND",    0, NULL, true, false, 0, NULL, 1, 1 , NULL);
  config_init_item(config, "FACIES_FOREGROUND",    0, NULL, true, false, 0, NULL, 1, -1, NULL);
  config_init_item(config, "NUM_GAUSS_FIELDS",     0, NULL, true, false, 0, NULL, 1, 1 , NULL);
  config_init_item(config, "TRUNCATION_SEQUENCE",  0, NULL, true, false, 0, NULL, 1, 1 , NULL);
  config_init_item(config, "PETROPHYSICS",         0, NULL, true, false, 0, NULL, 1, 1 , NULL);

  {
    int num_gauss_fields,i1, i2, j1,j2, k1,k2;
    config_item_type * config_item;

    config_parse(config, filename, ENKF_COM_KW);

    config_item = config_get_item(config, "COORDS");
    if(!util_sscanf_int(config_item_iget_argv(config_item,0),&i1))
      util_abort("%s: Failed to parse coordinate i1 from COORDS.\n",__func__);
    if(!util_sscanf_int(config_item_iget_argv(config_item,1),&i2))
      util_abort("%s: Failed to parse coordinate i2 from COORDS.\n",__func__);
    if(!util_sscanf_int(config_item_iget_argv(config_item,2),&j1))
      util_abort("%s: Failed to parse coordinate j1 from COORDS.\n",__func__);
    if(!util_sscanf_int(config_item_iget_argv(config_item,3),&j2))
      util_abort("%s: Failed to parse coordinate j2 from COORDS.\n",__func__);
    if(!util_sscanf_int(config_item_iget_argv(config_item,4),&k1))
      util_abort("%s: Failed to parse coordinate k1 from COORDS.\n",__func__);
    if(!util_sscanf_int(config_item_iget_argv(config_item,5),&k2))
      util_abort("%s: Failed to parse coordinate k2 from COORDS.\n",__func__);


    hash_insert_int(facies_kw_hash, config_get(config,"FACIES_BACKGROUND"),0);
    {
      int i;
      for(i=0; i<config_get_argc(config,"FACIES_FOREGROUND"); i++)
        hash_insert_int(facies_kw_hash,config_iget(config,"FACIES_FOREGROUND",i),i);
    }

    config_item = config_get_item(config,"NUM_GAUSS_FIELDS");
    if(!util_sscanf_int(config_item_iget_argv(config_item,0),&num_gauss_fields))
      util_abort("%s: Failed to parse integer from argument to NUM_GAUSS_FIELDS.\n",__func__);

    trunc_scheme = tpgzone_trunc_scheme_type_fscanf_alloc(config_get(config,"TRUNCATION_SEQUENCE"));
    petrophysics = tpgzone_config_petrophysics_fscanf_alloc(config_get(config,"PETROPHYSICS"));
    

  }

  
  config_free(config);
  return tpgzone_config;
}



/*
  Deallocator.
*/
void tpgzone_config_free(tpgzone_config_type * config)
{
  if(config->target_nodes != NULL) free(config->target_nodes);
  if(config->ecl_kw_name != NULL) free(config->ecl_kw_name);

  free(config);
}



/*****************************************************************/



/*
  Allocate a petrophysics model from a configuration file.
*/
scalar_config_type * tpgzone_config_petrophysics_fscanf_alloc(const char * filename)
{
  printf("%s: *WARNING* this function is empty.\n",__func__);
  return NULL;
}



/*
  Set petrophysics.
*/

void tpgzone_config_petrophysics_write(const tpgzone_config_type * config)
{
  int index;
  const int num_active_blocks = config->num_active_blocks;  
  double * target_buffer = util_malloc(num_active_blocks * ecl_util_get_sizeof_ctype(ecl_double_type),__func__);

  /*
    TODO

    Make this do something..
  */
  for(index = 0; index < num_active_blocks; index++)
  {
  }

  printf("%s: *WARNING* this function is empty.\n",__func__);

  free(target_buffer);
}

/*****************************************************************/

VOID_FREE(tpgzone_config)
GET_DATA_SIZE(tpgzone)
