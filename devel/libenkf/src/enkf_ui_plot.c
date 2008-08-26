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
}






void enkf_ui_plot_menu(void * _arg) {
  
  void_arg_type   * arg        = void_arg_safe_cast(_arg);
  enkf_main_type  * enkf_main  = void_arg_get_ptr(arg , 0);
  enkf_sched_type * enkf_sched = void_arg_get_ptr(arg , 1);

  menu_type * menu = menu_alloc("EnKF plot menu" , "qQ");
  menu_add_item(menu , "Test plot" , "tT" , simple_plot__ , NULL);
  menu_run(menu);
  menu_free(menu);

}
