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

  o PETROPHYSICS specs the petrophysics rule.
    The file will typically look something like:

   TARGET_FIELD_01 TARGET_FIELD_01_CONF_FILE 
   TARGET_FIELD_02 TARGET_FIELD_02_CONF_FILE 
   TARGET_FIELD_03 TARGET_FIELD_03_CONF_FILE 

    Here, TARGET_FIELD_* are arbitrary fields. E.g.,
    MY_PORO or MY_PERMZ_MULTIPLIER. The configuration
    files TARGET_FIELD_**_CONF_FILE should look like:

    SAND  UNIFORM   MIN MAX
    SHALE NORMAL    MU  STD 
    COAL  LOGNORMAL MU  STD

    Note that all facies *MUST* be present in
    the configuration file! The allowed transforms
    are equal to the transforms used e.g. in GEN_KW.
    

  That's about it.. 
*/



/*
  This function creates a truncated pluri-Gaussian zone based
  on the six indicies of a box.
*/
tpgzone_config_type * tpgzone_config_alloc_from_box(const ecl_grid_type       *  grid,
                                                    int                          num_gauss_fields,
                                                    int                          num_facies,
                                                    int                          num_target_fields,
                                                    char                     **  target_keys,
                                                    scalar_config_type       **  petrophysics,
                                                    tpgzone_trunc_scheme_type *  trunc_scheme,
                                                    hash_type                 *  facies_kw_hash,
                                                    int i1, int i2, int j1, int j2, int k1, int k2)
{
  tpgzone_config_type * config = util_malloc(sizeof * config,__func__);

  config->num_gauss_fields  = num_gauss_fields;
  config->num_facies        = num_facies;
  config->num_target_fields = num_target_fields;
  config->target_keys       = target_keys;
  config->petrophysics      = petrophysics;
  config->trunc_scheme      = trunc_scheme;
  config->facies_kw_hash    = facies_kw_hash;
  config->write_compressed  = true;
  config->ecl_kw_name       = NULL;

  /*
    FIXME

    This is just for testing, should not allow grid to be NULL
  */
  if(grid != NULL)
  {
    int nx,ny,nz,elements;
    ecl_box_type * box;
    
    if(grid == NULL)
      util_abort("%s: Internal error, grid pointer is not set.\n",__func__);

    ecl_grid_get_dims(grid,&nx,&ny,&nz, NULL);

    box                       = ecl_box_alloc(nx,ny,nz,i1,i2,j1,j2,k1,k2);
    elements                  = ecl_grid_count_box_active(grid,box);
    config->num_active_blocks = elements;

    ecl_grid_set_box_active_list(grid,box,config->target_nodes);

    config->data_size = elements * num_gauss_fields;

    ecl_box_free(box);
  }
  else
  {
    config->target_nodes = NULL;
  }
  return config;
}



tpgzone_config_type * tpgzone_config_fscanf_alloc(const char * conf_file, const ecl_grid_type * grid)
{
  tpgzone_config_type        * tpgzone_config  = NULL;
  hash_type                  * facies_kw_hash  = hash_alloc();
  tpgzone_trunc_scheme_type  * trunc_scheme    = NULL;
  scalar_config_type        ** petrophysics    = NULL;
  config_type                * config          = config_alloc(false);

  int num_gauss_fields, num_facies, num_target_fields;
  int i1, i2, j1,j2, k1,k2;

  char ** target_keys = NULL;

  config_init_item(config, "COORDS",               0, NULL, true, false, 0, NULL, 6, 6 , NULL);
  config_init_item(config, "FACIES_BACKGROUND",    0, NULL, true, false, 0, NULL, 1, 1 , NULL);
  config_init_item(config, "FACIES_FOREGROUND",    0, NULL, true, false, 0, NULL, 1, -1, NULL);
  config_init_item(config, "NUM_GAUSS_FIELDS",     0, NULL, true, false, 0, NULL, 1, 1 , NULL);
  config_init_item(config, "TRUNCATION_SEQUENCE",  0, NULL, true, false, 0, NULL, 1, 1 , NULL);
  config_init_item(config, "PETROPHYSICS",         0, NULL, true, false, 0, NULL, 1, 1 , NULL);

  {
    config_item_type * config_item;

    config_parse(config, conf_file, ENKF_COM_KW);

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
      util_abort("%s: failed to parse coordinate k1 from coords.\n",__func__);
    if(!util_sscanf_int(config_item_iget_argv(config_item,4),&k2))
      util_abort("%s: failed to parse coordinate k2 from coords.\n",__func__);


    hash_insert_int(facies_kw_hash, config_get(config,"FACIES_BACKGROUND"),0);
    {
      int i;
      num_facies = config_get_argc(config,"FACIES_FOREGROUND");
      for(i=0; i<num_facies; i++)
        hash_insert_int(facies_kw_hash,config_iget(config,"FACIES_FOREGROUND",i),i+1);

      num_facies++;
    }

    config_item = config_get_item(config,"NUM_GAUSS_FIELDS");
    if(!util_sscanf_int(config_item_iget_argv(config_item,0),&num_gauss_fields))
      util_abort("%s: Failed to parse integer from argument to NUM_GAUSS_FIELDS.\n",__func__);

    trunc_scheme = tpgzone_trunc_scheme_type_fscanf_alloc(config_get(config,"TRUNCATION_SEQUENCE"),
                                                                     facies_kw_hash,
                                                                     num_gauss_fields);

    tpgzone_config_petrophysics_fscanf_alloc(config_get(config,"PETROPHYSICS"),
                                             facies_kw_hash,
                                             &num_target_fields,
                                             &target_keys,
                                             &petrophysics);

  }

  tpgzone_config = tpgzone_config_alloc_from_box(grid, num_gauss_fields, num_facies,
                                                 num_target_fields, target_keys, petrophysics,
                                                 trunc_scheme, facies_kw_hash,
                                                 i1,i2,j1,j2,k1,k2);
  
  config_free(config);
  return tpgzone_config;
}



void tpgzone_config_free(tpgzone_config_type * config)
{
  if(config == NULL) util_abort("%s: Internal error, trying to free NULL pointer.\n",__func__);

  util_free_stringlist(config->target_keys,config->num_target_fields);

  {
    int i;
    for(i=0; i < config->num_target_fields; i++)
    {
      scalar_config_free(config->petrophysics[i]); 
    }

    free(config->petrophysics);
  }

  if(config->facies_kw_hash != NULL) hash_free(config->facies_kw_hash );
  if(config->target_nodes   != NULL)      free(config->target_nodes   );

  if(config->ecl_kw_name    != NULL)      free(config->ecl_kw_name    );
  
  

    
  /* TODO 
     Create code to free the truncation scheme.
  */

  free(config);
}



/*****************************************************************/


void tpgzone_config_petrophysics_fscanf_alloc(const char           * filename, 
                                              const hash_type      * facies_kw_hash,
                                              int                  * num_target_fields,
                                              char               *** __target_keys,
                                              scalar_config_type *** __petrophysics)
                                             
{
  config_type * config = config_alloc(true);
  config_parse(config, filename, ENKF_COM_KW);

  char ** target_keys;
  scalar_config_type ** petrophysics;
  target_keys = config_alloc_active_list(config, num_target_fields);
  petrophysics = util_malloc(*num_target_fields * sizeof  petrophysics,__func__);

  {
    int i;
    int num_facies = hash_get_size(facies_kw_hash);
    char *target_conf_filename;
    for(i=0; i<*num_target_fields; i++)
    {
     target_conf_filename = (char *) config_get(config,target_keys[i]);
     petrophysics[i] = tpgzone_config_petrophysics_fscanf_alloc_item(target_conf_filename, num_facies, facies_kw_hash);
    }
  }

  config_free(config);

  *__target_keys  = target_keys;
  *__petrophysics = petrophysics;
}



scalar_config_type * tpgzone_config_petrophysics_fscanf_alloc_item(const char        * filename,
                                                                   int                 num_facies,
                                                                   const hash_type   * facies_kw_hash)
{

  scalar_config_type * petrophysics_item;
  config_type *config = config_alloc(true);

  config_parse(config, filename, ENKF_COM_KW);

  {
    /*
      There must be a strict one-to-one relationship between the facies
      in the zone configuration and the facies in the petrophysics configuration.
      Thus, the first thing we do is to assert that this holds.
    */
    char ** facies_kw = hash_alloc_keylist( (hash_type *) facies_kw_hash);

    if(!config_has_keys(config, (const char **) facies_kw, num_facies, true))
    {
      int i,config_num_facies;
      char **config_facies_kw = config_alloc_active_list(config,&config_num_facies);

      printf("\n");
      printf("ERROR: Configuration error in TPGZONE.\n");
      printf("\n");
      printf("       I am asked to create a petrophysics model for the facies:\n");

      for(i=0; i<num_facies; i++)
      printf("       %02i:  %s\n",i,facies_kw[i]);

      printf("\n       However, the file %s contains petrophysics for the facies:\n",filename);
      for(i=0; i<config_num_facies; i++)
      printf("       %02i:  %s\n",i,config_facies_kw[i]);

      printf("\n");

      util_free_stringlist(config_facies_kw,config_num_facies);
      util_free_stringlist(facies_kw, num_facies);
      config_free(config);

      util_abort("%s: Error in petrophysics configuration - aborting.\n",__func__);
      return NULL;
    }
    util_free_stringlist(facies_kw, num_facies);
  }

  petrophysics_item = scalar_config_alloc_empty(num_facies);
  {
    int i, facies_int;
    FILE * stream;
    char facies_name[1024];
    stream = util_fopen(filename,"r");

    /*
     UGLY BUGLY: This does not support comments...
    */
    for(i=0; i<num_facies; i++)
    {
      if(fscanf(stream,"%s",facies_name) != 1)
        util_abort("%s: something wrong when reading: %s - aborting \n",__func__,filename);

      facies_int = hash_get_int(facies_kw_hash,facies_name);
      scalar_config_fscanf_line(petrophysics_item,facies_int,stream);
    }

    fclose(stream);
  }

  
  config_free(config);
  return petrophysics_item;
}



void tpgzone_config_petrophysics_write_field(const tpgzone_config_type * config, const double * data,
                                            const char * field_kw,  field_type * target_field)
{
  int field_num,grid_block_num;
  const int num_active_blocks = config->num_active_blocks;  
  double * target_buffer = util_malloc(num_active_blocks * ecl_util_get_sizeof_ctype(ecl_double_type),__func__);

  printf("%s: *WARNING* this function is empty.\n",__func__);

  free(target_buffer);
}



/*****************************************************************/



VOID_FREE(tpgzone_config)
GET_DATA_SIZE(tpgzone)
