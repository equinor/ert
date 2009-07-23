#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ecl_grid.h>
#include <ecl_kw.h>

#include "rms_file.h"
#include "rms_util.h"
#include "rms_export.h"


void rms_export_roff_from_keyword(const char *filename, ecl_grid_type *ecl_grid, 
    ecl_kw_type **ecl_kw, int size) {
  
  rms_file_type *rms_file;
  rms_tagkey_type *data_key;
  int nx, ny, nz, active_size;
  int i, j, k;
  int global_size;
  int n;
    
  ecl_grid_get_dims(ecl_grid, &nx, &ny, &nz, &active_size);
  global_size = ecl_grid_get_global_size(ecl_grid);
  
  rms_file = rms_file_alloc(filename, false);
  rms_file_fopen_w(rms_file);

  rms_file_init_fwrite(rms_file , "parameter");
  rms_tag_fwrite_dimensions(nx , ny , nz , rms_file_get_FILE(rms_file));

  for (n = 0; n < size; n++) {
    float *src_data;
    float *target_data;

    src_data = (float *) ecl_kw_get_void_ptr(ecl_kw[n]);
    target_data = util_malloc(global_size * sizeof(float), __func__);

    for (k=0; k < nz; k++) {
      for (j=0; j < ny; j++) {
        for (i=0; i < nx; i++) {
          int index1D;
          int index3D;
          double fill = RMS_INACTIVE_FLOAT;
          /* TODO:
           * This currently only supports FLOAT / REAL type.
           */ 
          
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
        rms_util_convert_ecl_type(ecl_kw_get_type(ecl_kw[n])), target_data, true);
    rms_tag_fwrite_parameter(ecl_kw_get_header_ref(ecl_kw[n]), data_key, 
        rms_file_get_FILE(rms_file));
    rms_tagkey_free(data_key);
    
    util_safe_free(target_data);
  }

  rms_file_complete_fwrite(rms_file);
  rms_file_fclose(rms_file);
  rms_file_free(rms_file);
  
}
