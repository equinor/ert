#include <hash.h>
#include <util.h>
#include <config.h>
#include <tpgzone.h>
#include <ecl_grid.h>
#include <trs.h>
#include <petp.h>



struct tpgzone_struct
{
  int          num_gauss_fields;
  char      ** gauss_field_files;

  char       * ecl_grid_file;
  int          num_blocks;
  int        * blocks;

  hash_type  * facies_kw_hash;
  
  trs_type   * trs;
  petp_type  * petp;
};


static int * tpgzone_alloc_blocks_from_grid(const char * grid_file, 
                                          int i1, int i2, int j1, int j2, int k1, int k2,
                                          int * num_blocks, bool endian_flip)
{
  int nx, ny, nz;
  int * blocks;
  ecl_box_type * ecl_box;

  ecl_grid_type * ecl_grid = ecl_grid_alloc(grid_file, endian_flip);
  ecl_grid_get_dims(ecl_grid,&nx,&ny,&nz, NULL);

  ecl_box    = ecl_box_alloc(nx, ny ,nz, i1, i2, j1, j2, k1, k2);
  *num_blocks = ecl_grid_count_box_active(ecl_grid, ecl_box);

  blocks = util_malloc(*num_blocks * sizeof * blocks, __func__);

  ecl_grid_set_box_active_list(ecl_grid, ecl_box, blocks);

  ecl_box_free(ecl_box);
  ecl_grid_free(ecl_grid);

  return blocks;
}



static double ** tpgzone_alloc_gauss_from_file(const tpgzone_type * tpgzone)
{
  double ** gauss = util_malloc(tpgzone->num_gauss_fields * sizeof * gauss, __func__);
}

/***************************************************************************************/



void tpgzone_summarize(const tpgzone_type * tpgzone)
{
  int i;

  printf("num_gauss_fields     : %d\n", tpgzone->num_gauss_fields);
  for(i=0; i<tpgzone->num_gauss_fields; i++)
  printf("gauss_field_files[%d] : %s\n", i, tpgzone->gauss_field_files[i]);

  printf("ecl_grid_file        : %s\n", tpgzone->ecl_grid_file);
  printf("num_blocks           : %d \n", tpgzone->num_blocks);
  printf("facies:\n");
  hash_printf_keys(tpgzone->facies_kw_hash);
  printf("blocks:\n");
  for(i=0; i<tpgzone->num_blocks; i++)
  printf("%d ",tpgzone->blocks[i]);
  printf("\n");

}



void tpgzone_free(tpgzone_type * tpgzone)
{
  free(tpgzone->ecl_grid_file);
  free(tpgzone->blocks       );

  util_free_stringlist(tpgzone->gauss_field_files, tpgzone->num_gauss_fields);

  hash_free(tpgzone->facies_kw_hash);
  trs_free( tpgzone->trs           );
  petp_free(tpgzone->petp          );

  free(tpgzone);
}



tpgzone_type * tpgzone_fscanf_alloc(char * config_file, bool endian_flip)
{
  tpgzone_type * tpgzone = util_malloc(sizeof * tpgzone, __func__);

  printf("Loading tpgzone info from %s...\n", config_file);
  {

    config_type * config = config_alloc(false);

    config_init_item(config, "COORDS",               0, NULL, true, false, 0, NULL, 6, 6 , NULL);
    config_init_item(config, "FACIES_BACKGROUND",    0, NULL, true, false, 0, NULL, 1, 1 , NULL);
    config_init_item(config, "FACIES_FOREGROUND",    0, NULL, true, false, 0, NULL, 1, -1, NULL);
    config_init_item(config, "GAUSS_FIELD_FILES",    0, NULL, true, false, 0, NULL, 1, -1, NULL);
    config_init_item(config, "ECL_GRID_FILE"    ,    0, NULL, true, false, 0, NULL, 1, 1,  NULL);
    config_init_item(config, "TRUNCATION_SEQUENCE",  0, NULL, true, false, 0, NULL, 1, 1 , NULL);
    config_init_item(config, "PETROPHYSICS",         0, NULL, true, false, 0, NULL, 1, 1 , NULL);

    config_parse(config, config_file, ENKF_COM_KW);



    {
      /*
        Parsing of ecl_grid_file, num_blocks and blocks. 
      */

      int    i1, i2, j1, j2, k1, k2;
      int    num_blocks;
      int  * blocks;
      char * ecl_grid_file;

      if(!util_sscanf_int(config_iget(config, "COORDS", 0) ,&i1))
        util_abort("%s: failed to parse coordinate i1 from COORDS.\n",__func__);

      if(!util_sscanf_int(config_iget(config, "COORDS", 1) ,&i2))
        util_abort("%s: failed to parse coordinate i2 from COORDS.\n",__func__);

      if(!util_sscanf_int(config_iget(config, "COORDS", 2) ,&j1))
        util_abort("%s: failed to parse coordinate j1 from COORDS.\n",__func__);

      if(!util_sscanf_int(config_iget(config, "COORDS", 3) ,&j2))
        util_abort("%s: failed to parse coordinate j2 from COORDS.\n",__func__);

      if(!util_sscanf_int(config_iget(config, "COORDS", 4) ,&k1))
        util_abort("%s: failed to parse coordinate k1 from COORDS.\n",__func__);

      if(!util_sscanf_int(config_iget(config, "COORDS", 5) ,&k2))
        util_abort("%s: failed to parse coordinate k2 from COORDS.\n",__func__);

      
      ecl_grid_file = util_alloc_string_copy(config_get(config, "ECL_GRID_FILE"));
      blocks    = tpgzone_alloc_blocks_from_grid(ecl_grid_file, i1, i2, j1, j2, k1, k2, &num_blocks, endian_flip);

      tpgzone->ecl_grid_file = ecl_grid_file;
      tpgzone->num_blocks    = num_blocks;
      tpgzone->blocks        = blocks;
    }
    
    {
      /* 
        Parsing of facies labels. 
       */

      int i;
      int num_foreground_facies = config_get_argc(config,"FACIES_FOREGROUND");

      hash_type * facies_kw_hash = hash_alloc();
      hash_insert_int(facies_kw_hash, config_get(config,"FACIES_BACKGROUND"), 0);

      for(i=0; i<num_foreground_facies; i++)
        hash_insert_int(facies_kw_hash,config_iget(config,"FACIES_FOREGROUND", i),i+1);

      tpgzone->facies_kw_hash = facies_kw_hash;
    }

    {
      /*
        Parsing of gauss_field_files.
      */

      int i, num_gauss_fields;
      char ** gauss_field_files;

      num_gauss_fields = config_get_argc(config, "GAUSS_FIELD_FILES");

      gauss_field_files = util_malloc(num_gauss_fields * sizeof * gauss_field_files, __func__);

      for(i=0; i<num_gauss_fields; i++)
        gauss_field_files[i] = util_alloc_string_copy(config_iget(config, "GAUSS_FIELD_FILES", i));
        
      tpgzone->num_gauss_fields  = num_gauss_fields;
      tpgzone->gauss_field_files = gauss_field_files;
    }

    {
      /*
        Parsing of truncation scheme.
      */

      const char * trs_config_filename = config_get(config, "TRUNCATION_SEQUENCE");
      trs_type   * trs = trs_fscanf_alloc(trs_config_filename, tpgzone->facies_kw_hash, tpgzone->num_gauss_fields);

      tpgzone->trs = trs;
    }

    {
      /*
        Parsing of petrophysics.
      */
      const char * petp_config_filename = config_get(config, "PETROPHYSICS");
      petp_type  * petp = petp_fscanf_alloc(petp_config_filename, tpgzone->facies_kw_hash);
      
      tpgzone->petp = petp;
    }

    config_free(config);
  }
  tpgzone_summarize(tpgzone);
  return tpgzone;
}


void tpgzone_apply(const tpgzone_type * tpgzone)
{
  double ** gauss = tpgzone_alloc_gauss_from_file(tpgzone);
  
  {
    int i;
    for(i=0; i<tpgzone->num_gauss_fields; i++)
      free(gauss[i]);
  } 
  free(gauss);
}
