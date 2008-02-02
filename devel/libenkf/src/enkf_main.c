#include <enkf_fs.h>
#include <enkf_ensemble.h>
#include <util.h>
#include <plain_driver.h>
#include <config.h>
#include <hash.h>


int main (int argc , char ** argv) {
  const char * data_file     = "GRANE_orig.DATA";
  /*const char * run_path      = "/d/felles/bg/scratch/EnKF_Grane2008/Testing/tmpdir_%04d";*/
  const char * run_path      = "/tmp/enkf/RunPATH/tmpdir_%04d";
  const char * eclbase       = "GRANE-%04d";
  const char * schedule_file = "SCHEDULE_orig.INC";
  const int start_date[3]    = { 1 , 1 , 1977};
  sched_file_type *s;

  {
    hash_type * config_hash = hash_alloc(10);
    config_parse("Config/main" , config_hash);
  }
  
  s = sched_file_alloc(start_date);
  sched_file_parse(s , schedule_file);
  plain_driver_type * dynamic_analyzed = plain_driver_alloc("/tmp/enkf/Ensemble/%04d/mem%03d/Analyzed");
  plain_driver_type * dynamic_forecast = plain_driver_alloc("/tmp/enkf/Ensemble/%04d/mem%03d/Forecast");
  plain_driver_type * eclipse_static   = plain_driver_alloc("/tmp/enkf/Ensemble/%04d/mem%03d/Static");
  plain_driver_type * parameter        = plain_driver_alloc("/tmp/enkf/Ensemble/%04d/mem%03d/Parameter");
  
  enkf_fs_type  * fs = enkf_fs_alloc(dynamic_analyzed, dynamic_forecast , eclipse_static , parameter);
  enkf_ensemble_type * enkf_ensemble;
  enkf_ensemble = enkf_ensemble_alloc(1 , fs , data_file , run_path , eclbase , s , false , false , true);

  enkf_ensemble_add_well(enkf_ensemble , "PR10_G18" , 4 , (const char *[4]) {"WGPR" , "WWPR" , "WOPR" , "WBHP"});
  enkf_ensemble_add_well(enkf_ensemble , "T21A"     , 4 , (const char *[4]) {"WGPR" , "WWPR" , "WOPR" , "WBHP"});
  enkf_ensemble_add_well(enkf_ensemble , "PR03A_G8" , 4 , (const char *[4]) {"WGPR" , "WWCT" , "WOPR" , "WBHP"});
  enkf_ensemble_add_well_obs(enkf_ensemble , "PR03A_G8" , NULL , "Config/PRO3A_G8");
  enkf_ensemble_add_gen_kw(enkf_ensemble , "Config/gen_kw_config.txt");
  
  /*
    enkf_ensemble_load_ecl_init_mt(enkf_ensemble , 307);
    {
    int iens;
    for (iens = 0; iens < 20; iens++)
    enkf_ensemble_iload_ecl_mt(enkf_ensemble , iens);
    
    }
    enkf_ensemble_load_ecl_complete_mt(enkf_ensemble);
    enkf_ensemble_analysis(enkf_ensemble);
  */
  
  enkf_ensemble_add_data_kw(enkf_ensemble , "INIT" , "INCLUDE\n  \'EQUIL.INC\'/\n");
  enkf_ensemble_init_eclipse(enkf_ensemble);

  sched_file_free(s);
  enkf_ensemble_free(enkf_ensemble);
  enkf_fs_free(fs);
}
