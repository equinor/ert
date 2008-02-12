#include <enkf_fs.h>
#include <enkf_main.h>
#include <enkf_config.h>
#include <util.h>
#include <plain_driver.h>
#include <plain_driver_static.h>
#include <plain_driver_parameter.h>
#include <config.h>
#include <hash.h>
#include <fs_index.h>
#include <enkf_types.h>


int main (int argc , char ** argv) {
  const char * data_file     = "GRANE_orig.DATA";
  const char * run_path      = "/d/felles/bg/scratch/Testing/tmpdir_%04d";
  const char * eclbase       = "GRANE-%04d";
  const char * schedule_file = "SCHEDULE_orig.INC";
  const char * grid_file     = "/d/proj/bg/enkf/EnKF_Grane2008/Refcase/GRANE.EGRID";
  
  {
    hash_type * config_hash = hash_alloc(10);
    config_parse("Config/main" , config_hash);
  }
  
  /* 
     It is not clear whether we should send in a filename - or a
     completey parsed object?

     Currently we send in complete objects - and they are owned in
     this scope.
  */
  
  
  enkf_main_type   * enkf_main;
  enkf_config_type * enkf_config = enkf_config_fscanf_alloc("Config/enkf" , 1 , false , false , true);

  plain_driver_type * dynamic_analyzed 	      = plain_driver_alloc(enkf_config_get_ens_path(enkf_config) 	   , "%04d/mem%03d/Analyzed");
  plain_driver_type * dynamic_forecast 	      = plain_driver_alloc(enkf_config_get_ens_path(enkf_config) 	   , "%04d/mem%03d/Forecast");
  plain_driver_parameter_type * parameter     = plain_driver_parameter_alloc(enkf_config_get_ens_path(enkf_config) , "%04d/mem%03d/Parameter");
  plain_driver_static_type * eclipse_static   = plain_driver_static_alloc(enkf_config_get_ens_path(enkf_config)    , "%04d/mem%03d/Static");
  fs_index_type     * fs_index                = fs_index_alloc("./Ensemble/mem%03d/INDEX");
  
  enkf_fs_type     * fs = enkf_fs_alloc(fs_index , dynamic_analyzed, dynamic_forecast , eclipse_static , parameter);
  enkf_main = enkf_main_alloc(enkf_config , fs);

  
  /*
  enkf_config_add_type(config , "SWAT"  , ecl_restart , FIELD , 
		       field_config_alloc("SWAT" , ecl_float_type   , nx , ny , nz , active_size , index_map , 1));
  
  enkf_config_add_type(config , "PRESSURE" , ecl_restart , FIELD , 
		       field_config_alloc("PRESSURE"  , ecl_float_type , nx , ny , nz , active_size , index_map , 1));
		       
  enkf_config_add_type(config , "SGAS"  , ecl_restart , FIELD , 
		       field_config_alloc("SGAS" , ecl_float_type    , nx , ny , nz , active_size , index_map , 1 ));


  enkf_config_add_type(config , "RS"     , ecl_restart , FIELD , 
		       field_config_alloc("RS"  , ecl_float_type        , nx , ny , nz , active_size , index_map , 1 ));
  
  enkf_config_add_type(config , "RV"    , ecl_restart , FIELD , 
		       field_config_alloc("RV"    , ecl_float_type       , nx , ny , nz , active_size , index_map , 1 ));
  */


  enkf_config_add_gen_kw(enkf_config , "Config/gen_kw_config.txt");
  
  enkf_main_load_ecl_init_mt(enkf_main , 319);
  {
    int iens;
    for (iens = 0; iens < 10; iens++)
      enkf_main_iload_ecl_mt(enkf_main , iens);
    
  }
  enkf_main_load_ecl_complete_mt(enkf_main);
  enkf_main_analysis(enkf_main);
  exit(1);
  
  enkf_main_add_data_kw(enkf_main , "INIT" , "INCLUDE\n  \'EQUIL.INC\'/\n");
  enkf_main_init_eclipse(enkf_main);
  
  enkf_main_free(enkf_main);
  enkf_fs_free(fs);
}
