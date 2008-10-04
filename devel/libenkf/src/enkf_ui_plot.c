#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <util.h>
#include <ctype.h>
#include <menu.h>
#include <void_arg.h>
#include <enkf_main.h>
#include <enkf_sched.h>
#include <enkf_ui_plot.h>
#include <plot.h>
#include <plot_dataset.h>
#include <enkf_ui_util.h>
#include <ensemble_config.h>


void simple_plot__(void *p) {
  plot_type *item;
  const double period = 2 * PI;
  
  item = plot_alloc();
  plot_set_window_size(item, 640, 480);
  plot_initialize(item, "png", "test.png");
  
  {
    plot_dataset_type *d;
    int N = pow(2, 10);
    PLFLT x[2 * N];
    PLFLT y[2 * N];
    int i;
    
    for (i = 0; i <= 2 * N; i++) {
      x[i] = (i - N) / period;
      if (x[i] != 0.0)
	y[i] = sin(PI * x[i]) / (PI * x[i]);
      else
	y[i] = 1.0;
    }
    d = plot_dataset_alloc();
    plot_dataset_set_data(d, x, y, 2 * N, BLUE, LINE);
    plot_dataset_add(item, d);
  }
  
  plot_set_labels(item, "x-axis", "y-axis", "y = sinc(x)", BLACK);
  plot_set_viewport(item, -period, period, -0.3, 1);
  plot_data(item);
  plot_free(item);

  util_vfork_exec("/d/proj/bg/enkf/bin/qiv" , 1 , (const char *[1]) {"test.png"} , false , NULL , NULL , NULL , NULL , NULL);
}



void stupid_plot(int N , const double * x , const double *y) {
  plot_type *item;
  
  item = plot_alloc();
  plot_set_window_size(item, 640, 480);
  plot_initialize(item, "png", "test.png");
  
  {
    plot_dataset_type *d = plot_dataset_alloc();
    plot_dataset_set_data(d, x, y, 2 * N, BLUE, LINE);
    plot_dataset_add(item, d);
  }
  
  plot_set_labels(item, "x-axis", "y-axis", "y = sinc(x)", BLACK);
  {
    double xmin,xmax;
    double ymin,ymax;
    
    util_double_vector_max_min(N , x , &xmax , &xmin);
    util_double_vector_max_min(N , y , &ymax , &ymin);
    plot_set_viewport(item, xmin , xmax , ymin , ymax);
  }
  plot_data(item);
  plot_free(item);
  
  util_vfork_exec("/d/proj/bg/enkf/bin/qiv" , 1 , (const char *[1]) {"test.png"} , false , NULL , NULL , NULL , NULL , NULL);
}





void enkf_ui_plot_time(void *_arg) {
  void_arg_type   * arg    = void_arg_safe_cast(_arg);
  enkf_main_type  * enkf_main   = void_arg_get_ptr(arg , 0);
  enkf_sched_type * enkf_sched  = void_arg_get_ptr(arg , 1);
  const ensemble_config_type * ensemble_config = enkf_main_get_ensemble_config(enkf_main);
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
	stupid_plot(size , x , y);
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
  
  menu_type * menu = menu_alloc("EnKF plot menu" , "qQ");
  menu_add_item(menu , "Plot time development" , "tT" , enkf_ui_plot_time , arg);
  menu_run(menu);
  menu_free(menu);

}
