#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <arg_pack.h>
#include <enkf_main.h>
#include <enkf_sched.h>
#include <enkf_ui_plot.h>
#include <plot.h>
#include <plot_dataset.h>
#include <enkf_ui_util.h>
#include <ensemble_config.h>
#include <msg.h>




static plot_type * __plot_alloc(const char * x_label , const char * y_label , const char * title , const char * file) {
  plot_type * plot  = plot_alloc();
  plot_set_window_size(plot , 640, 480);
  plot_initialize(plot , "png", file);
  plot_set_labels(plot, x_label , y_label , title , BLACK);
  return plot;
}


static void __plot_add_data(plot_type * plot , int N , const double * x , const double *y, bool first) {
  plot_dataset_type *d = plot_dataset_alloc( false , false);
  plot_dataset_set_data(d, x, y, N, BLUE, LINE);
  plot_dataset_add(plot, d);
}


static void __plot_show(plot_type * plot , const char * viewer , const char * file) {
  plot_set_viewport( plot );
  plot_data(plot);
  plot_free(plot);
  util_vfork_exec(viewer , 1 , (const char *[1]) { file } , false , NULL , NULL , NULL , NULL , NULL);
}


void stupid_plot(int N , const double * x , const double *y , const char * image_viewer) {
  plot_type *item;
  
  item = plot_alloc();
  plot_set_window_size(item, 640, 480);
  plot_initialize(item, "png", "test.png");
  
  {
    plot_dataset_type *d = plot_dataset_alloc( false , false );
    plot_dataset_set_data(d, x, y, N, BLUE, LINE);
    plot_dataset_add(item, d);
  }
  
  plot_set_labels(item, "x-axis", "y-axis", "y = sinc(x)", BLACK);
  plot_set_viewport(item);
  plot_data(item);
  plot_free(item);
  
  util_vfork_exec(image_viewer , 1 , (const char *[1]) {"test.png"} , false , NULL , NULL , NULL , NULL , NULL);
}





void enkf_ui_plot_ensemble(void * arg_pack) {
  enkf_main_type             * enkf_main  = enkf_main_safe_cast( arg_pack );
  const enkf_sched_type      * enkf_sched = enkf_main_get_enkf_sched(enkf_main);
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const char * viewer                          = enkf_main_get_image_viewer( enkf_main );
  {
    const int prompt_len = 40;
    const char * prompt  = "What do you want to plot (KEY:INDEX)";
    const enkf_config_node_type * config_node;
    state_enum analysis_state;
    int        cell_nr;
    int        size;
    char      *plot_file;
    char      *key_index;
    char       user_key[64];
    
    
    util_printf_prompt(prompt , prompt_len , '=' , "=> ");
    scanf("%s" , user_key);
    plot_file = util_alloc_sprintf("/tmp/%s.png" , user_key);
    {
      plot_type * plot = __plot_alloc("x-akse","y-akse",user_key,plot_file);
      bool first = true;
      msg_type * msg;
      state_enum              plot_state;
      const int last_report = enkf_sched_get_last_report(enkf_sched);
      int iens1 , iens2 , step1 , step2;   
      double * x, *y;
      int iens , step; /* Observe that iens and report_step loops below should be inclusive.*/
      enkf_node_type * node;
      enkf_fs_type   * fs   = enkf_main_get_fs(enkf_main);

      config_node = ensemble_config_user_get_node( ensemble_config , user_key , &key_index);
      if (config_node == NULL) {
	fprintf(stderr,"** Sorry - could not find any nodes with the key:%s \n",user_key);
	plot_free(plot);
	return;
      }

      enkf_ui_util_scanf_report_steps(last_report , prompt_len , &step1 , &step2);
      enkf_ui_util_scanf_iens_range(ensemble_config_get_size(ensemble_config) , prompt_len , &iens1 , &iens2);
	
      node = enkf_node_alloc( config_node );
      {
	enkf_var_type var_type = enkf_config_node_get_var_type(config_node);
	if (var_type == dynamic)
	  plot_state = both;
	else if (var_type == parameter)
	  plot_state = analyzed;
	else
	  util_abort("%s: can not plot this type \n",__func__);
      }
      if (plot_state == both) 
	size = 2 * (step2 - step1 + 1);
      else
	size = (step2 - step1 + 1);

      x = util_malloc( size * sizeof * x, __func__);
      y = util_malloc( size * sizeof * y, __func__);
      msg = msg_alloc("Loading member/step: ");
      msg_show(msg);
      for (iens = iens1; iens <= iens2; iens++) {
	char label[32];

	int this_size = 0;
	for (step = step1; step <= step2; step++) {
	  sprintf(label , "%03d/%03d" , iens , step);
	  msg_update( msg , label);
	  /* Skipping forecast. */
	  if (enkf_fs_has_node(fs , config_node , step , iens , analyzed)) {
	    bool valid;
	    enkf_fs_fread_node(fs , node , step , iens , analyzed);
	    y[this_size] = enkf_node_user_get( node , key_index , &valid);
	    if (valid) {
	      x[this_size] = step;
	      this_size++;
	    }
	  }
	}
	/*stupid_plot(this_size , x , y , viewer);*/
	__plot_add_data(plot , this_size , x , y , first);
	first = false;
      }
      msg_free(msg , true);
      printf("Plot saved in: %s \n",plot_file);
      __plot_show(plot , viewer , plot_file);
      free(plot_file);
    }
  }
}
		   
	  


void enkf_ui_plot_time(void * arg_pack) {
  enkf_main_type             * enkf_main  = enkf_main_safe_cast( arg_pack );
  const enkf_sched_type      * enkf_sched = enkf_main_get_enkf_sched(enkf_main);
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
  const char * viewer                          = enkf_main_get_image_viewer( enkf_main );
  {
    const int prompt_len = 35;
    const enkf_config_node_type * config_node;
    state_enum analysis_state;
    int        cell_nr;
    int        size;
    
    config_node = enkf_ui_util_scanf_parameter(ensemble_config , prompt_len , true , FIELD , invalid , NULL , &analysis_state , NULL);
    cell_nr = enkf_ui_util_scanf_ijk(enkf_config_node_get_ref(config_node) , prompt_len);
    {
      const int last_report = enkf_sched_get_last_report(enkf_sched);
      const int step1       = util_scanf_int_with_limits("First report step",prompt_len , 0 , last_report);
      const int step2       = util_scanf_int_with_limits("Last report step",prompt_len , step1 , last_report);
      int iens1 , iens2;   
      bool * iens_active    = enkf_ui_util_scanf_alloc_iens_active( ensemble_config_get_size(ensemble_config) , prompt_len , &iens1 , &iens2); /* Not used yet ... */
      double * x, *y;
      int iens; /* Observe that iens and report_step loops below should be inclusive.*/
      enkf_node_type * node = enkf_node_alloc( config_node );
      enkf_fs_type   * fs   = enkf_main_get_fs(enkf_main);
      path_fmt_type * file_fmt = path_fmt_scanf_alloc("Give filename to store line (with %d for report iens) =>" , 0 , NULL , false);
      
      
      if (analysis_state == both) 
	size = 2 * (step2 - step1 + 1);
      else
	size = (step2 - step1 + 1);

      x = util_malloc( size * sizeof * x, __func__);
      y = util_malloc( size * sizeof * y, __func__);
      
      for (iens = iens1; iens <= iens2; iens++) {
	enkf_ui_util_get_time(fs , config_node , node , analysis_state , cell_nr , step1 , step2 , iens , x ,y);
	stupid_plot(size , x , y , viewer);
	{
	  char * filename = path_fmt_alloc_file(file_fmt , true , iens);
	  FILE * stream = util_fopen(filename , "w");
	  int    index  = 0;
	  int    report_step;
	  for (report_step = step1; report_step <= step2; report_step++) {
	    fprintf(stream , "%g  %g \n",x[index] , y[index]);
	    index++;
	    if (analysis_state == both) {
	      fprintf(stream , "%g  %g \n",x[index] , y[index]);
	      index++;
	    }
	  }
	  fclose(stream);
	  free(filename);
	}
      }
    }
  }
}



	



void enkf_ui_plot_menu(void * arg) {

  enkf_main_type  * enkf_main  = enkf_main_safe_cast( arg );  
  menu_type * menu = menu_alloc("EnKF plot menu" , "qQ");

  menu_add_item(menu , "Ensemble plot" , "eE" , enkf_ui_plot_ensemble , enkf_main );
  menu_run(menu);
  menu_free(menu);

}
