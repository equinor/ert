#include <stdlib.h>
#include <util.h>
#include <plplot/plplot.h>
#include <plot_driver.h>


typedef struct {
  int stream;
} plplot_state;


static void plplot_set_range( void * driver , double xmin , double xmax , double ymin , double ymax) {
  plwind(xmin,xmax,ymin,ymax);
}


static void plplot_free_driver( void * driver ) {
  return ; 
}


static void plplot_set_window_size( void * driver , int width , int height) {
  char * geometry = util_alloc_sprintf("%dx%d", width, height);
  plsetopt("geometry", geometry);
  free(geometry);
}



static void plplot_plot_points( void * driver , point_attribute_type * point_attr , int num_points , const double * x , const double * y) {
  plcol0(point_attr->point_color);
  plpoin(num_points , (double *) x , (double *) y , point_attr->symbol_type);
}



static void plplot_plot_line( void * driver , line_attribute_type * line_attr , int num_points , const double * x , const double * y) {
  if (num_points == 1) {
    plcol0(line_attr->line_color);
    plpoin(num_points , (double *) x , (double *) y , PLOT_DEFAULT_SYMBOL);
  } else {
    plcol0(line_attr->line_color);
    pllsty(line_attr->line_style);         /* Setting solid/dashed/... */
    plline(num_points , (double *) x , (double *) y);
  }
}




plot_driver_type * plplot_driver_alloc() {
  plot_driver_type * driver = plot_driver_alloc_empty(PLPLOT);
  
  driver->set_range   	  = plplot_set_range;
  driver->free_driver 	  = plplot_free_driver;
  driver->set_window_size = plplot_set_window_size;
  driver->plot_points     = plplot_plot_points;
  driver->plot_line       = plplot_plot_line;
  
  return driver;
}
