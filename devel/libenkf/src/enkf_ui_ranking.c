#include <stdlib.h>
#include <util.h>
#include <stdio.h>
#include <enkf_main.h>
#include <menu.h>
#include <enkf_obs.h>




static void enkf_ui_ranking_ALL( void * arg) {
  const state_enum load_state = both;
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  enkf_fs_type               * fs              = enkf_main_get_fs(enkf_main);
  enkf_obs_type              * enkf_obs        = enkf_main_get_obs( enkf_main );
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const int ens_size = ensemble_config_get_size(ensemble_config);
  double * chi2 = util_malloc( ens_size * sizeof * chi2 , __func__);
  int iens;

  //enkf_obs_total_ensemble_chi2( enkf_obs , fs , ens_size , load_state , chi2);
  //enkf_obs_ensemble_chi2( enkf_obs , fs , 30 , ens_size , load_state , chi2);

  printf("\n");
  printf(" ----------------------------------\n");
  printf(" Realization  |  Total chi^2 misfit\n");
  printf(" ----------------------------------\n");
  for (iens = 0; iens < ens_size; iens++)
    printf(" %11d  |       %g \n",iens , chi2[iens]);
  printf(" ----------------------------------\n\n");
  free( chi2 );
}




void enkf_ui_ranking_menu(void * arg) {
  
  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  {
    menu_type * menu = menu_alloc("Ranking of results" , "Back" , "bB");
    menu_add_item(menu , "All observation / all timesteps"                    , "1"  , enkf_ui_ranking_ALL   , enkf_main , NULL);
    menu_run(menu);
    menu_free(menu);
  }
}
