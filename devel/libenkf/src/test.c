#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <gen_data.h>
#include <gen_data_config.h>
#include <util.h>
#include <ecl_grid.h>
#include <pilot_point_config.h>
#include <pilot_point.h>
#include <forward_model.h>
#include <rms_file.h>
#include <rms_util.h>

#define GRID_FILE "/d/felles/bg/scratch/masar/STRUCT/realization-24-step-0-to-358/RG01-STRUCT-24.EGRID"
#define RESTART_FILE "/d/felles/bg/scratch/masar/STRUCT/realization-24-step-0-to-358/RG01-STRUCT-24.X0050"
#define RMS_OUT "rms.roff"

void ecl_rms_export_keywords(const char *roff_filename, ecl_grid_type *ecl_grid, ecl_kw_type **ecl_kw) {
  
}


int main(void) {
  ecl_grid_type *ecl_grid;
  ecl_file_type *ecl_file;
  ecl_kw_type *ecl_kw;
  rms_file_type *rms_file;
  rms_tagkey_type *data_key;
  int nx, ny, nz, active_size;
  int i, j, k;
  float *src_data;
  float *target_data;
  int global_size;
    

  ecl_grid = ecl_grid_alloc(GRID_FILE);
  ecl_grid_get_dims(ecl_grid, &nx, &ny, &nz, &active_size);
  printf("dims: %d, %d, %d\n", nx, ny, nz);
  global_size = ecl_grid_get_global_size(ecl_grid);
  printf("glob: %d\n", global_size);

  ecl_file = ecl_file_fread_alloc (RESTART_FILE);
  ecl_kw = ecl_file_iget_named_kw (ecl_file, "PRESSURE", 0);

  src_data = (float *) ecl_kw_get_void_ptr(ecl_kw);
  
  rms_file = rms_file_alloc(RMS_OUT, false);
  rms_file_fopen_w(rms_file);
  rms_file_init_fwrite(rms_file , "parameter");
  rms_tag_fwrite_dimensions(nx , ny , nz , rms_file_get_FILE(rms_file));

  target_data = util_malloc(global_size * sizeof(float), __func__);

  for (k=0; k < nz; k++) {
    for (j=0; j < ny; j++) {
      for (i=0; i < nx; i++) {
        int index1D;
        int index3D;
        double fill = RMS_INACTIVE_FLOAT;
        
        index1D = ecl_grid_get_active_index3(ecl_grid, i, j, k);
        index3D = rms_util_global_index_from_eclipse_ijk(nx, ny, nz, i, j, k);

        if (index1D >= 0)
          target_data[index3D] = src_data[index1D];
        else
          memcpy(&target_data[index3D] , &fill, sizeof(float));
      }
    }
  }

  data_key = rms_tagkey_alloc_complete("data", global_size, 
      rms_util_convert_ecl_type(ecl_kw_get_type(ecl_kw)), target_data, true);
  rms_tag_fwrite_parameter("MARTIN", data_key, 
      rms_file_get_FILE(rms_file));
  rms_tagkey_free(data_key);

  rms_file_complete_fwrite(rms_file);
  rms_file_fclose(rms_file);
  rms_file_free(rms_file);
  
  util_safe_free(target_data);

  ecl_file_free(ecl_file);
  ecl_grid_free(ecl_grid);
}



