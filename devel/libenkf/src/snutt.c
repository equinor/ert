#include <util.h>
#include <ecl_grid.h>
#include <field_config.h>
#include <field.h>

/* Oldskoool fortran shit from libsample. Don't touch. */
extern void m_pseudo2d_mp_pseudo2d_(double *A, const int * nx, const int * ny,
                                    const int * lde, const double * rx, const double * ry,
                                    const double * dx, const double * dy, const int * n1,
                                    const int * n2, const double * theta, const int * ver);
 


static double * sample_gauss_field_alloc(int nx, int ny, int nz, double rng, double std)
{
  /* For compatibility with fortran... sigh.. */
  double df0 = 0.0;
  double df1 = 1.0;
  int    if1 = 1;
  int    fverbose = util_C2f90_bool(false);

  double * field = util_malloc(nx * ny * sizeof * field, __func__);

  if(nz > 1)
    util_abort("%s: Sorry, no support for nz > 1.\n", __func__);

  m_pseudo2d_mp_pseudo2d_(field, &nx, &ny, &if1, &rng, &rng, &df1, &df1,
                          &nx, &ny, &df0, &fverbose);
  
  for(int i=0; i<nx*ny; i++)
    field[i] = std * field[i];

  return field;
}



static void ar_update(double * inc, double * src, int n,  double rho)
{
  if(rho < 0 || rho > 1.0)
    util_abort("%s: Error, need rho in [0,1]. Aborting. \n");
  for(int i=0; i<n; i++)
  {
    src[i] = (1-rho) * src[i] + rho * inc[i];
  }
}



static void vec_add(double * inc, double * src, int n)
{
  for(int i=0; i<n; i++)
    src[i] = src[i] + inc[i];
}



int main(int argc, char * argv[])
{
  if(argc < 7)
  {
    printf("Usage: snutt.x ecl_grid_file ecl_restart_file mod_err_file std rng rho.\n");
    return 0;
  }



  bool   endian_flip = true;
  char * ecl_grid_file     = argv[1];
  char * ecl_restart_file  = argv[2];
  char * mod_err_file      = argv[3];

  double std, rng, rho;
  double * mod_err_dbl;
  double * mod_err_dbl_inc;
  double * pres_dbl;

  int nx, ny, nz, active_size;
  int * index_map;


  ecl_grid_type * ecl_grid = ecl_grid_alloc(ecl_grid_file, endian_flip);
  index_map = ecl_grid_get_index_map_ref(ecl_grid);
  ecl_grid_get_dims(ecl_grid, &nx, &ny, &nz, &active_size);


  {
    if(!util_sscanf_double(argv[4], &std)) 
      util_abort("%s: Could'nt allocate a double from %s.\n", __func__, argv[4]);
    if(!util_sscanf_double(argv[5], &rng)) 
      util_abort("%s: Could'nt allocate a double from %s.\n", __func__, argv[5]);
    if(!util_sscanf_double(argv[6], &rho)) 
      util_abort("%s: Could'nt allocate a double from %s.\n", __func__, argv[6]);
    if(nz > 1)
      util_abort("%s: No support for nz > 1. Aborting.\n", __func__);
    if(nx*ny != active_size)
      util_abort("%s: No support for grids with inactive blocks. Aborting\n", __func__);
  }


  if(util_file_exists(mod_err_file))
  {
    /*
      Load and update.
    */
    field_config_type * mod_err_config = field_config_alloc_parameter_no_init("PMERR", ecl_grid);
    field_type        * mod_err        = field_alloc(mod_err_config);
    field_fload_auto(mod_err, mod_err_file, endian_flip);

    mod_err_dbl     = field_indexed_get_alloc(mod_err, active_size, index_map);
    mod_err_dbl_inc = sample_gauss_field_alloc(nx, ny, nz, rng, std);
    ar_update(mod_err_dbl_inc, mod_err_dbl, active_size, rho);

    field_indexed_set(mod_err, ecl_double_type, active_size, index_map, mod_err_dbl);
    field_export(mod_err, mod_err_file, ecl_kw_file_all_cells);

    free(mod_err_dbl_inc);
    free(mod_err);
    free(mod_err_config);
  }
  else
  {
    /*
      Initialize and exit.
    */
    field_config_type * mod_err_config = field_config_alloc_parameter_no_init("PMERR", ecl_grid);
    field_type        * mod_err        = field_alloc(mod_err_config);

    mod_err_dbl = sample_gauss_field_alloc(nx, ny, nz, rng, std);

    field_indexed_set(mod_err, ecl_double_type, active_size, index_map, mod_err_dbl);
    field_export(mod_err, mod_err_file, ecl_kw_file_all_cells);

    free(mod_err_dbl);
    free(mod_err);
    free(mod_err_config);
    
    ecl_grid_free(ecl_grid);
    return 0;
  }


  field_config_type * pres_config = field_config_alloc_dynamic("PRESSURE", ecl_grid);
  field_type * pres = field_alloc(pres_config);
  field_fload_auto(pres, ecl_restart_file, endian_flip);

  pres_dbl = field_indexed_get_alloc(pres, active_size, index_map);
  vec_add(mod_err_dbl, pres_dbl, active_size);
  field_indexed_set(pres, ecl_double_type, active_size, index_map, pres_dbl);

  field_export(pres, ecl_restart_file, ecl_kw_file_all_cells);

  free(mod_err_dbl);
  free(pres_dbl);
  field_free(pres);
  field_config_free(pres_config);
  ecl_grid_free(ecl_grid);

}
