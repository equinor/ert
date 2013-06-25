/*
   Copyright (C) 2012  Statoil ASA, Norway. 
    
   The file 'enkf_tui_QC.c' is part of ERT - Ensemble based Reservoir Tool. 
    
   ERT is free software: you can redistribute it and/or modify 
   it under the terms of the GNU General Public License as published by 
   the Free Software Foundation, either version 3 of the License, or 
   (at your option) any later version. 
    
   ERT is distributed in the hope that it will be useful, but WITHOUT ANY 
   WARRANTY; without even the implied warranty of MERCHANTABILITY or 
   FITNESS FOR A PARTICULAR PURPOSE.   
    
   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html> 
   for more details. 
*/
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>

#include <ert/util/double_vector.h>
#include <ert/util/util.h>
#include <ert/util/menu.h>
#include <ert/util/arg_pack.h>
#include <ert/util/path_fmt.h>
#include <ert/util/bool_vector.h>
#include <ert/util/msg.h>
#include <ert/util/vector.h>

#include <ert/plot/plot.h>
#include <ert/plot/plot_dataset.h> 

#include <ert/ecl/ecl_rft_file.h>

#include <ert/enkf/enkf_main.h>
#include <ert/enkf/enkf_obs.h>
#include <ert/enkf/block_obs.h>
#include <ert/enkf/gen_obs.h>
#include <ert/enkf/field_config.h>
#include <ert/enkf/obs_vector.h>
#include <ert/enkf/ensemble_config.h>
#include <ert/enkf/enkf_state.h>
#include <ert/enkf/gen_kw_config.h>
#include <ert/enkf/enkf_defaults.h>
#include <ert/enkf/plot_config.h>
#include <ert/enkf/member_config.h>
#include <ert/enkf/enkf_analysis.h>
#include <ert/enkf/obs_tstep_list.h>

#include <enkf_tui_util.h>
#include <enkf_tui_plot.h>
#include <enkf_tui_fs.h>
#include <ert_tui_const.h>
#include <enkf_tui_plot_util.h>


void enkf_tui_QC_plot_get_PC( enkf_main_type * enkf_main , int step1 , int step2 , state_enum state , const local_obsset_type * obsset , 
                              double truncation , int ncomp , 
                              matrix_type * PC , matrix_type * PC_obs) {
  
  int               ens_size             = enkf_main_get_ensemble_size( enkf_main );
  obs_data_type  *  obs_data             = obs_data_alloc();
  meas_data_type *  meas_data            = meas_data_alloc( ens_size );
  analysis_config_type * analysis_config = enkf_main_get_analysis_config( enkf_main );
  int_vector_type * step_list            = int_vector_alloc(0,0);    

  {
    for (int step =step1; step <= step2; step++) 
      int_vector_append( step_list , step );
  }
  
  obs_data_reset( obs_data );
  meas_data_reset( meas_data );

  {
    double std_cutoff = analysis_config_get_std_cutoff( analysis_config );
    double alpha      = analysis_config_get_alpha( analysis_config );

    enkf_obs_get_obs_and_measure(enkf_main_get_obs( enkf_main ),
                                 enkf_main_get_fs( enkf_main ), 
                                 step_list , 
                                 state, 
                                 ens_size,
                                 (const enkf_state_type **) enkf_main_get_ensemble( enkf_main ),
                                 meas_data , 
                                 obs_data , 
                                 obsset );
  
    enkf_analysis_deactivate_outliers( obs_data , meas_data  , std_cutoff , alpha);
  }
  
  {
    int active_size      = obs_data_get_active_size( obs_data );
    matrix_type * S      = meas_data_allocS( meas_data , active_size );
    matrix_type * dObs   = obs_data_allocdObs( obs_data , active_size );

    obs_data_scale( obs_data , S , NULL , NULL , NULL , dObs );
    enkf_main_get_PC( S , dObs , truncation , ncomp , PC , PC_obs );
    
    matrix_free( S );
    matrix_free( dObs );
  }

  int_vector_free( step_list );
  obs_data_free( obs_data );
  meas_data_free( meas_data );
}



void enkf_tui_QC_plot_PC( void * arg ) {
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  const int last_report                  = enkf_main_get_history_length( enkf_main );
  int step1,step2;
  double truncation_or_ncomp;
  local_obsdata_type * obsdata = local_obsdata_alloc();
  
  

  enkf_tui_util_scanf_report_steps(last_report , PROMPT_LEN , &step1 , &step2);
  util_printf_prompt("Truncation: [0,1): Explained variance  [1,ens_size): fixed" , PROMPT_LEN , '=' , "=> ");
  {
    char * input = util_alloc_stdin_line();

    if (input == NULL)
      return;
    else {
      if (!util_sscanf_double( input , &truncation_or_ncomp)) {
        fprintf(stderr , "Failed to parse:%s as number \n",input);
        free( input );
        return;
      }
    }
        
    free( input );
  }
  
  {
    stringlist_type * obs_keys = enkf_obs_alloc_keylist( enkf_main_get_obs( enkf_main ));
    int iobs;
    for (iobs = 0; iobs < stringlist_get_size( obs_keys); iobs++) {
      const char * obs_key = stringlist_iget( obs_keys , iobs );
      local_obsdata_node_type * obs_node = local_obsdata_node_alloc( obs_key );

      local_obsdata_node_add_range( obs_node , step1 , step2 );
      local_obsdata_add_node( obsdata , obs_node );
    }
    

    stringlist_free( obs_keys );
  }  
    
  {
    matrix_type * PC     = matrix_alloc(1,1);
    matrix_type * PC_obs = matrix_alloc(1,1);
    analysis_config_type * analysis_config = enkf_main_get_analysis_config( enkf_main );
    char * plot_name = util_alloc_sprintf(analysis_config_get_PC_filename( analysis_config ) , 
                                          step1 , step2 , "obs");
    
    enkf_main_init_PC( enkf_main , obsdata , truncation_or_ncomp , PC , PC_obs);
    enkf_tui_plot_PC( enkf_main , plot_name , PC , PC_obs );
    
    free( plot_name );
    matrix_free( PC );
    matrix_free( PC_obs );
    local_obsdata_free( obsdata );
  }
}


void enkf_tui_QC_run_workflow( void * arg ) {
  enkf_main_type  * enkf_main        = enkf_main_safe_cast( arg );  
  const qc_module_type  * qc_module  = enkf_main_get_qc_module( enkf_main );
  
  qc_module_run_workflow( qc_module , enkf_main );
}



void enkf_tui_QC_menu(void * arg) {
  
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  plot_config_type * plot_config = enkf_main_get_plot_config( enkf_main );
  {
    const char * plot_path  =  plot_config_get_path( plot_config );
    util_make_path( plot_path );
  }
  
  {
    menu_type * menu = menu_alloc("Quality check of prior" , "Back" , "bB");
    menu_item_type * plot_PC_item         = menu_add_item( menu , "Plot of prior principal components"    , "pP"  , enkf_tui_QC_plot_PC , enkf_main , NULL);
    menu_item_type * run_QC_workflow_item = menu_add_item( menu , "Run QC workflow"    , "rR"  , enkf_tui_QC_run_workflow , enkf_main , NULL);
    
    if (!enkf_main_have_obs( enkf_main )) 
      menu_item_disable( plot_PC_item );
    
    if (!enkf_main_has_QC_workflow( enkf_main ))
      menu_item_disable( run_QC_workflow_item );
    
    menu_run(menu);
    menu_free(menu);
  }
}

