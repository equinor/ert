#include <util.h>
#include <ecl_grid.h>
#include <tpgzone_util.h>



field_config_type * tpgzone_field_config_alloc__
                    (const char * ecl_filename,
                     const char * ecl_kw,
                     const ecl_grid_type * ecl_grid,
                     bool endian_flip
                     )
{
  field_config_type * field_config;
  field_file_type file_type = field_config_guess_file_type(ecl_filename, endian_flip);
  if(file_type == unknown_file)
    util_abort("%s: Sorry, ecl_kw, ecl_grdecl or rms_roff files are supported - aborting.\n", __func__);

  {
    int nx, ny, nz, active_size;
    ecl_grid_get_dims(ecl_grid, &nx, &ny, &nz, &active_size);

    const int *index_map = ecl_grid_get_index_map_ref(ecl_grid);

    field_config = field_config_alloc_parameter_no_init(ecl_kw,
                                              nx, ny, nz, active_size,
                                              index_map);

    if(file_type == ecl_grdecl_file);
      field_config->ecl_export_format = ecl_grdecl_format;
  }

  return field_config;
}
