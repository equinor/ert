{
   const ecl_grid_type * ecl_grid = ecl_config_get_grid( enkf_main->ecl_config );
   local_reportstep_type * reportstep = local_config_alloc_reportstep( enkf_main->local_config , "DEFAULT");
   
/* Northern part */
{ 
  const int i1 = 0;
  const int i2 = 101;
  const int j1 = 115;
  const int j2 = 165;
  const int k1 = 0;
  const int k2 = 19;
  local_ministep_type * ministep_north = local_config_alloc_ministep( enkf_main->local_config , "NORTH");
  
  local_ministep_add_obs( ministep_north , "WWPR:PR01_G1");
  local_ministep_add_obs( ministep_north , "WWPR:PR02_G12");
  local_ministep_add_obs( ministep_north , "WWPR:PR06_G28");
  local_ministep_add_obs( ministep_north , "WWPR:PR07_G02");
  local_ministep_add_obs( ministep_north , "WWPR:PR03A_G8");

  local_ministep_add_obs( ministep_north , "WGPR:PR01_G1");
  local_ministep_add_obs( ministep_north , "WGPR:PR02_G12");
  local_ministep_add_obs( ministep_north , "WGPR:PR06_G28");
  local_ministep_add_obs( ministep_north , "WGPR:PR07_G02");
  local_ministep_add_obs( ministep_north , "WGPR:PR03A_G8");


  {
    int i,j,k;
    local_ministep_add_node( ministep_north , "PORO");
    local_ministep_add_node( ministep_north , "PERMX");
    local_ministep_add_node( ministep_north , "PRESSURE");

    active_list_type * active_poro  = local_ministep_get_node_active_list( ministep_north , "PORO");
    active_list_type * active_permx = local_ministep_get_node_active_list( ministep_north , "PERMX");
    active_list_type * active_pres  = local_ministep_get_node_active_list( ministep_north , "PRESSURE");
    
    for (i = i1; i <= i2; i++)
      for (j = j1; j <= j2; j++)
	for (k = k1; k <= k2; k++) {
	  int active_index = ecl_grid_get_active_index3( ecl_grid , i,j,k);
	  if (active_index >= 0) {
	    /* i,j,k er en aktiv celle */
	    active_list_add_index( active_poro  , active_index );
	    active_list_add_index( active_permx , active_index );
	    active_list_add_index( active_pres  , active_index );
	  }
	}
  }
  local_reportstep_add_ministep( reportstep , ministep_north);
}


/* Middle part */
{ 
  const int i1 = 0;
  const int i2 = 101;
  const int j1 = 82;
  const int j2 = 114;
  const int k1 = 0;
  const int k2 = 19;
  local_ministep_type * ministep_middle = local_config_alloc_ministep( enkf_main->local_config , "MIDDLE");
  
  local_ministep_add_obs( ministep_middle , "WWPR:PR11E_G5");
  local_ministep_add_obs( ministep_middle , "WWPR:PR12_G19");
  local_ministep_add_obs( ministep_middle , "WWPR:PR13_G22");
  local_ministep_add_obs( ministep_middle , "WWPR:PR24_G17");
  local_ministep_add_obs( ministep_middle , "WWPR:PR26_G26");
  local_ministep_add_obs( ministep_middle , "WWPR:PR11_G6");
  local_ministep_add_obs( ministep_middle , "WWPR:PR10_G18");
  local_ministep_add_obs( ministep_middle , "WWPR:PR09_G10");

  local_ministep_add_obs( ministep_middle , "WGPR:PR11E_G5");
  local_ministep_add_obs( ministep_middle , "WGPR:PR12_G19");
  local_ministep_add_obs( ministep_middle , "WGPR:PR13_G22");
  local_ministep_add_obs( ministep_middle , "WGPR:PR24_G17");
  local_ministep_add_obs( ministep_middle , "WGPR:PR26_G26");
  local_ministep_add_obs( ministep_middle , "WGPR:PR11_G6");
  local_ministep_add_obs( ministep_middle , "WGPR:PR10_G18");
  local_ministep_add_obs( ministep_middle , "WGPR:PR09_G10");

  {
    int i,j,k;
    local_ministep_add_node( ministep_middle , "PORO");
    local_ministep_add_node( ministep_middle , "PERMX");
    local_ministep_add_node( ministep_middle , "PRESSURE");

    active_list_type * active_poro  = local_ministep_get_node_active_list( ministep_middle , "PORO");
    active_list_type * active_permx = local_ministep_get_node_active_list( ministep_middle , "PERMX");
    active_list_type * active_pres  = local_ministep_get_node_active_list( ministep_middle , "PRESSURE");
    
    for (i = i1; i <= i2; i++)
      for (j = j1; j <= j2; j++)
	for (k = k1; k <= k2; k++) {
	  int active_index = ecl_grid_get_active_index3( ecl_grid , i,j,k);
	  if (active_index >= 0) {
	    /* i,j,k er en aktiv celle */
	    active_list_add_index( active_poro  , active_index );
	    active_list_add_index( active_permx , active_index );
	    active_list_add_index( active_pres  , active_index );
	  }
	}
  }
  local_reportstep_add_ministep( reportstep , ministep_middle);
}

/* South part */
{ 
  const int i1 = 0;
  const int i2 = 101;
  const int j1 = 0;
  const int j2 = 81;
  const int k1 = 0;
  const int k2 = 19;
  local_ministep_type * ministep_south = local_config_alloc_ministep( enkf_main->local_config , "SOUTH");
  
  local_ministep_add_obs( ministep_south , "WWPR:PR25_G21");
  local_ministep_add_obs( ministep_south , "WWPR:PR27_G11");
  local_ministep_add_obs( ministep_south , "WWPR:PR23_G9");
  local_ministep_add_obs( ministep_south , "WWPR:PR22_G25");
  local_ministep_add_obs( ministep_south , "WWPR:PR18_G40");
  local_ministep_add_obs( ministep_south , "WWPR:PR16_G15");
  local_ministep_add_obs( ministep_south , "WWPR:PR17_G30");
  local_ministep_add_obs( ministep_south , "WWPR:PR15_G35");
  local_ministep_add_obs( ministep_south , "WWPR:PR29_G34");
  local_ministep_add_obs( ministep_south , "WWPR:PR20_G39");

  local_ministep_add_obs( ministep_south , "WGPR:PR25_G21");
  local_ministep_add_obs( ministep_south , "WGPR:PR27_G11");
  local_ministep_add_obs( ministep_south , "WGPR:PR23_G9");
  local_ministep_add_obs( ministep_south , "WGPR:PR22_G25");
  local_ministep_add_obs( ministep_south , "WGPR:PR18_G40");
  local_ministep_add_obs( ministep_south , "WGPR:PR16_G15");
  local_ministep_add_obs( ministep_south , "WGPR:PR17_G30");
  local_ministep_add_obs( ministep_south , "WGPR:PR15_G35");
  local_ministep_add_obs( ministep_south , "WGPR:PR29_G34");
  local_ministep_add_obs( ministep_south , "WGPR:PR20_G39");


  {
    int i,j,k;
    local_ministep_add_node( ministep_south , "PORO");
    local_ministep_add_node( ministep_south , "PERMX");
    local_ministep_add_node( ministep_south , "PRESSURE");

    active_list_type * active_poro  = local_ministep_get_node_active_list( ministep_south , "PORO");
    active_list_type * active_permx = local_ministep_get_node_active_list( ministep_south , "PERMX");
    active_list_type * active_pres  = local_ministep_get_node_active_list( ministep_south , "PRESSURE");
    
    for (i = i1; i <= i2; i++)
      for (j = j1; j <= j2; j++)
	for (k = k1; k <= k2; k++) {
	  int active_index = ecl_grid_get_active_index3( ecl_grid , i,j,k);
	  if (active_index >= 0) {
	    /* i,j,k er en aktiv celle */
	    active_list_add_index( active_poro  , active_index );
	    active_list_add_index( active_permx , active_index );
	    active_list_add_index( active_pres  , active_index );
	  }
	}
  }
  local_reportstep_add_ministep( reportstep , ministep_south);
}









local_config_set_default_reportstep( enkf_main->local_config , "DEFAULT");
}
