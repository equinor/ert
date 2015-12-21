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
#include <ert/util/int_vector.h>
#include <ert/util/util.h>
#include <ert/util/menu.h>
#include <ert/util/arg_pack.h>
#include <ert/util/path_fmt.h>
#include <ert/util/bool_vector.h>
#include <ert/util/msg.h>
#include <ert/util/vector.h>
#include <ert/util/matrix.h>
#include <ert/util/type_vector_functions.h>


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
#include <ert/enkf/pca_plot_data.h>

#include <enkf_tui_util.h>
#include <enkf_tui_fs.h>
#include <ert_tui_const.h>





void enkf_tui_QC_run_workflow( void * arg ) {
  enkf_main_type  * enkf_main        = enkf_main_safe_cast( arg );
  const hook_manager_type  * hook_manager  = enkf_main_get_hook_manager( enkf_main );

  hook_manager_run_workflow( hook_manager , enkf_main );
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
    menu_item_type * run_QC_workflow_item = menu_add_item( menu , "Run QC workflow"    , "rR"  , enkf_tui_QC_run_workflow , enkf_main , NULL);
    
    if (!enkf_main_has_QC_workflow( enkf_main ))
      menu_item_disable( run_QC_workflow_item );

    menu_run(menu);
    menu_free(menu);
  }
}

