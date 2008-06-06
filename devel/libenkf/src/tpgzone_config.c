#include <ecl_grid.h>
#include <tpgzone_config.h>
#include <util.h>

/*
  This function creates a truncated pluri-Gaussian zone based
  on the six indicies of a box.
*/
tpgzone_config_type * tpgzone_config_alloc_from_box(const ecl_grid_type       * grid,
                                                    int                         num_gauss_fields,
                                                    int                         num_facies,
                                                    tpgzone_trunc_scheme_type * trunc_scheme,
                                                    const void_arg_type       * trunc_arg,
                                                    hash_type                 * facies_kw,
                                                    scalar_config_type        * poro,
                                                    scalar_config_type        * permx,
                                                    scalar_config_type        * permz,
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
  config->facies_kw        = facies_kw;
  config->poro             = poro,
  config->permx            = permx,
  config->permz            = permz,
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

    /*
      Set the active nodes
    */
    ecl_grid_set_box_active_list(grid,box,config->target_nodes);

    /*
      From the macro CONFIG_STD_FIELDS.
      
      Note that there are currently only three parameters pr. facies.
    */
    config->data_size = 3 * num_facies + elements * num_gauss_fields;

    /*
      Free allocated memory.
    */
    ecl_box_free(box);
  }
  return config;
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


/*****************************************************************/

VOID_FREE(tpgzone_config)
GET_DATA_SIZE(tpgzone)
