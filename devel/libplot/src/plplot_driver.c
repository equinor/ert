#include <stdlib.h>
#include <util.h>
#include <plot_driver.h>
#include <plplot/plplot.h>


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



plot_driver_type * plplot_driver_alloc() {
  plot_driver_type * driver = plot_driver_alloc_empty(PLPLOT);
  
  driver->set_range   	  = plplot_set_range;
  driver->free_driver 	  = plplot_free_driver;
  driver->set_window_size = plplot_set_window_size;
  
  return driver;
}
