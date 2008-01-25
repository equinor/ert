#include <stdlib.h>
#include <enkf_util.h>
#include <enkf_state.h>
#include <enkf_node.h>
#include <enkf_config.h>
#include <multz_config.h>
#include <multz.h>
#include <equil_config.h>
#include <equil.h>
#include <ecl_kw.h>
#include <ecl_block.h>
#include <enkf_types.h>
#include <field_config.h>
#include <well_config.h>
#include <thread_pool.h>
#include <void_arg.h>
#include <well_obs.h>
#include <history.h>
#include <sched_file.h>
#include <obs_data.h>
#include <meas_vector.h>
#include <enkf_obs.h>
#include <multflt_config.h>
#include <multflt.h>
#include <equil_config.h>
#include <trans_func.h>
#include <ecl_grid.h>
#include <pgbox_config.h>
#include <enkf_fs.h>
#include <plain_driver.h>


double pgfilter(const double *data , const void_arg_type * arg) {
  double x = data[0];
  double y = data[1];

  return x + y;
}



void TEST() {
  return ; 
}



int main(void) {

  enkf_fs_type       * enkf_fs;
  enkf_obs_type      * enkf_obs;
  obs_data_type      * obs_data;
  meas_vector_type     * meas_vector;
  enkf_config_type   * config;
  enkf_state_type   ** state;

  const int *index_map;
  int nx,ny,nz,active_size;
  thread_pool_type * tp;

  sched_file_type    * sched;
  history_type          * hist;
  ecl_grid_type * grid      = ecl_grid_alloc_EGRID("ECLIPSE.EGRID" , true);
  meas_vector = meas_vector_alloc();
  obs_data  = obs_data_alloc();

  TEST();
  sched     = sched_file_alloc((const int [3]) {1 , 1 , 1999});
  sched_file_parse(sched , "SCHEDULE.INC");
  hist      = history_alloc_from_schedule(sched);
  index_map = ecl_grid_alloc_index_map(grid);
  ecl_grid_get_dims(grid , &nx , &ny , &nz , &active_size);
  

  enkf_fs = enkf_fs_alloc( plain_driver_alloc("path1") , plain_driver_alloc("path2") , plain_driver_alloc("path3") , plain_driver_alloc("path4"));

  config = enkf_config_alloc("RunPATH/tmpdir_%04d" , "Ensemble/%04d/Static/mem%04d" , "Ensemble/%04d/Parameters/mem%04d" , "Ensemble/%04d/Dynamic/Forecast/mem%04d" , "Ensemble/%04d/Dynamic/Analyzed/mem%04d" , true);
  enkf_config_add_type(config , "MULTZ" , 
		       parameter , MULTZ, 
		       multz_config_fscanf_alloc("Config/multz" , 100 , 100 , 100));
  
  enkf_config_add_type(config , "EQUIL" , 
		       parameter , EQUIL, 
		       equil_config_fscanf_alloc("Config/equil"));
  
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

  {
    field_config_type * permx_config = field_config_alloc("PERMX" , ecl_float_type , nx , ny , nz , active_size , index_map , 1 );
    
    enkf_config_add_type(config , "PERMX" , parameter , FIELD , permx_config);  /* This is released by the config object */
    enkf_config_add_type(config , "PG"    , parameter , PGBOX , pgbox_config_alloc(grid , pgfilter , NULL , 2 , 1 , nx , 1 , ny , 1 , 10 , permx_config));
  }

  
  enkf_config_add_well(config , "B-37T2" ,  4 , (const char *[4]) {"WGPR" , "WWPR" , "WOPR" , "WBHP"});
  enkf_config_add_well(config , "B-33A"  ,  4 , (const char *[4]) {"WGPR" , "WWPR" , "WOPR" , "WBHP"});
  enkf_config_add_well(config , "B-43A"  ,  4 , (const char *[4]) {"WGPR" , "WWPR" , "WOPR" , "WBHP"});

  enkf_obs = enkf_obs_alloc(config , sched);
  enkf_obs_add_well_obs(enkf_obs , "B-33A"  , NULL, "Config/B-33A");
  enkf_obs_add_well_obs(enkf_obs , "B-37T2" , NULL, "Config/B-43A");
  enkf_obs_add_well_obs(enkf_obs , "B-43A"  , NULL, "Config/B-37T2");
  
  /*
    enkf_obs_add_field_obs(enkf_obs , "PRES"  , 4 , (const char *[4]) {"WGPR" , "WWPR" , "WOPR" , "WBHP"});
  */



  tp = thread_pool_alloc(10);
  state = malloc(100 * sizeof *state);
  {
    void_arg_type ** load_arg = calloc(100 , sizeof * load_arg);
    int i;
    
    for (i=0; i < 100; i++) {
      state[i] = enkf_state_alloc(config , "ECLIPSE" , i , NULL , false);
      enkf_state_add_node(state[i] , "MULTZ"); 
      enkf_state_add_node(state[i] , "EQUIL");
      enkf_state_add_node(state[i] , "PERMX");
      enkf_state_add_node(state[i] , "PG");
      {
	bool unified    = false;
	int report_step = 51;

	load_arg[i] = void_arg_alloc3(pointer_value , bool_value , int_value);
	
	void_arg_pack_ptr(load_arg[i]  , 0 , state[i]);
	void_arg_pack_bool(load_arg[i] , 1 , unified);
	void_arg_pack_int(load_arg[i]  , 2 , report_step);
	
	thread_pool_add_job(tp , &enkf_state_load_ecl_summary_void , load_arg[i]);
	thread_pool_add_job(tp , &enkf_state_load_ecl_restart_void , load_arg[i]);
      }
    }
    thread_pool_join(tp);
    for (i=0; i < 100; i++) 
      void_arg_free(load_arg[i]);

    {
      int i;
      double *X = calloc(100 * 100 , sizeof *X);
      for (i=0; i < 100; i++)
	X[i*100 + i] = 1.0;
      enkf_ensemble_update(state , 100 , 100 * 100000 , X); 
      free(X);
    }
    enkf_obs_get_observations(enkf_obs , 51 , obs_data);
    obs_data_fprintf(obs_data   , stdout);
    for (i = 0; i < 100; i++) 
      enkf_obs_measure(enkf_obs   , 51 , state[i] , meas_vector);
    meas_vector_fprintf(meas_vector , stdout);
    

    for (i=0; i < 100; i++)
      enkf_state_free(state[i]);
    
  }    
  free((int *) index_map);
  ecl_grid_free(grid);
  obs_data_reset(obs_data);
  obs_data_fprintf(obs_data , stdout);
  enkf_config_free(config);
  enkf_obs_free(enkf_obs);
  history_free(hist);
  sched_file_free(sched);
}
