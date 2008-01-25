#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ecl_block.h>
#include <ecl_kw.h>
#include <util.h>
#include <ecl_util.h>
#include <fortio.h>
#include <field_config.h>
#include <field.h>
#include <ecl_grid.h>

const char * add_NOSIM_exe = "/d/proj/bg/enkf/bin/add_NOSIM";
const char * del_NOSIM_exe = "/d/proj/bg/enkf/bin/del_NOSIM";
const char * submit_exe    = "/d/proj/bg/enkf/bin/ecl_submit.x";



void doit(const char * path, const char * base_name , const char *swcr_file , const char * sowcr_file) {
  const bool endian_convert = true;
  const bool fmt_file       = false;
  char * restart_file = ecl_util_alloc_filename(path , base_name , ecl_restart_file , fmt_file , 0);
  if (!util_file_exists(restart_file)) {
    fprintf(stderr,"Can not find restart file: %s - aborting \n",restart_file);
    exit(1);
  }

  {
    int nx,ny,nz,active_size;
    char * EGRID_file             = ecl_util_alloc_filename(path , base_name , ecl_egrid_file  , fmt_file , 0);
    ecl_grid_type * ecl_grid      = ecl_grid_alloc_EGRID(EGRID_file , true);
    const int * index_map         = ecl_grid_alloc_index_map(ecl_grid);
    ecl_grid_get_dims(ecl_grid , &nx , &ny , &nz , &active_size);
    
    ecl_kw_type    * sowcr_kw;    
    ecl_kw_type    * swcr_kw;
    ecl_kw_type    * swat_kw;
    ecl_kw_type    * sgas_kw;
    ecl_kw_type    * soil_kw;
    fortio_type    * fortio;
  
    fortio = fortio_open(restart_file , "r" , endian_convert);
    ecl_kw_fseek_kw("SWAT" , false , true , true , fortio);
    swat_kw = ecl_kw_fread_alloc(fortio , false);
    
    fortio_rewind(fortio);
    ecl_kw_fseek_kw("SGAS" , false , true , true , fortio);
    sgas_kw = ecl_kw_fread_alloc(fortio , false);
    fortio_close(fortio);

    soil_kw = ecl_kw_alloc_copy(swat_kw);
    ecl_kw_scalar_init(soil_kw , 1.0);
    ecl_kw_inplace_sub(soil_kw , sgas_kw);
    ecl_kw_inplace_sub(soil_kw , swat_kw);

    /*****************************************************************/

    swcr_kw  = ecl_kw_alloc_copy(swat_kw);   /* SWCR = SWAT(t = 0)                  */

    sowcr_kw = ecl_kw_alloc_copy(swcr_kw);   /* SOWCR = 1 / (4 + (1 / (1 - SWCR)))  */
    ecl_kw_scale(sowcr_kw , -1);
    ecl_kw_shift(sowcr_kw ,  1);
    ecl_kw_inplace_inv(sowcr_kw);
    ecl_kw_shift(sowcr_kw , 4);
    ecl_kw_inplace_inv(sowcr_kw);

    ecl_kw_fwrite_param(swcr_file  , false , true , "SWCR"  , ecl_float_type , nx*ny*(51 - 27 + 1) , ecl_kw_iget_ptr(swcr_kw , nx*ny*26));
    ecl_kw_fwrite_param(sowcr_file , false , true , "SOWCR" , ecl_float_type , nx*ny*(51 - 27 + 1) , ecl_kw_iget_ptr(soil_kw , nx*ny*26));

    free((int *) index_map);
    free(EGRID_file);
    ecl_kw_free(swcr_kw);
    ecl_kw_free(swat_kw);
    ecl_kw_free(sgas_kw);
    ecl_kw_free(soil_kw);
  }
  free(restart_file);
}



int main( int argc , char ** argv) {

  if (argc == 1) {
    fprintf(stderr,"%s: BASENAME - aborting \n",argv[0]);
    exit(1);
  }
  {
    const char * swcr_file  = "SWCR";
    const char * sowcr_file = "SOWCR";
    char * base_name = argv[1];
    char * data_file = malloc(strlen(base_name) + 6);
    char * run_path  = "./";
    char   cmd[256];
    FILE * stream;

    sprintf(data_file , "%s.DATA" , base_name);
    if (util_file_exists(data_file)) {
      sprintf(cmd , "%s %s" , add_NOSIM_exe , data_file);
      system(cmd);
      
      stream = fopen("SWCR-INIT" , "w");
      fclose(stream);
      
      printf("Starting NOSIM ECLIPSE simulation \n");
      sprintf(cmd , "%s %s %s 0" , submit_exe , run_path , base_name);
      system(cmd);
      
      doit(run_path , base_name , swcr_file , sowcr_file);
      
      stream = fopen("SWCR-INIT" , "w");
      fprintf(stream , "IMPORT\n   '%s'   /\n" , swcr_file);
      fprintf(stream , "IMPORT\n   '%s'   /\n" , sowcr_file);
      fclose(stream);
      
      sprintf(cmd , "%s %s" , del_NOSIM_exe , data_file);
      system(cmd);
    } else {
      fprintf(stderr,"%s: data file:%s does not exist - aborting \n",argv[0], data_file);
      abort();
    }
  }      
  return 0;
}
