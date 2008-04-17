

struct havana_fault_config_struct {
  const gen_kw_config_struct * gen_kw_config;
  char  *havana_executable;
}



/*****/

"/tmp/enkf/run_eclipse/Case3/tmpdir_0004/MULTZ.INC"

void havana_fault_ecl_write(const havana_fault_type * havana_fault, const char * ecl_file) {
  char * run_path;
  char * havana_model_file;
  char * cmd;
  
  util_alloc_file_components(ecl_file , &run_path , NULL , NULL);
  havana_model_file = util_alloc_full_path(run_path , "havan.model");
  cmd = util_alloc_joined_string( (const char *[2]) {havana_fault_config_get_exe(havana_fault->config)  , havana_mdel_file} , 2 , " ");

  gen_kw_ecl_write( havana_fault->gen_kw , havana_model_file);
  system(cmd);
  /* Kopipere resultat inn i run_path */
  util_copy_file(havana_output_file , ecl_file);
  

  free(run_path);
  free(cmd);
  free(havana_model_file);
}
