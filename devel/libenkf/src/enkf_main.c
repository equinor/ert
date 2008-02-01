#include <enkf_fs.h>
#include <enkf_ens.h>
#include <plain_driver.h>


int main (int argc , char ** argv) {
  const char * run_path = "tmp_link/tmpdir_%04d";
  const char * eclbase  = "GRANE-%04d";
  const char * schedule_file = "SCHEDULE_orig.INC";
  const int start_date[3] = { 1 , 1 , 1977};

  plain_driver_type * dynamic_analyzed = plain_driver_alloc("Ensemble/%04d/mem%03d/Analyzed");
  plain_driver_type * dynamic_forecast = plain_driver_alloc("Ensemble/%04d/mem%03d/Forecast");
  plain_driver_type * eclipse_static   = plain_driver_alloc("Ensemble/%04d/mem%03d/Static");
  plain_driver_type * parameter        = plain_driver_alloc("Ensemble/%04d/mem%03d/Parameter");
  
  enkf_fs_type  * fs = enkf_fs_alloc(dynamic_analyzed, dynamic_forecast , eclipse_static , parameter);
  enkf_ens_type * enkf_ens;
  
  enkf_ens = enkf_ens_alloc(100 , fs , run_path , eclbase , schedule_file , start_date , false , true);

  enkf_ens_free(enkf_ens);
}
