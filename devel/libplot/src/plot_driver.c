#include <plot_driver.h>
#include <plot_const.h>
#include <util.h>

plot_driver_type * plot_driver_alloc_empty(plot_driver_enum driver_id) {
  plot_driver_type * driver = util_malloc(sizeof * driver , __func__);
  driver->driver_id   	  = driver_id;
  driver->state       	  = NULL;	       
  driver->set_range   	  = NULL;      
  driver->set_title   	  = NULL;      
  driver->set_xlabel  	  = NULL;     
  driver->set_ylabel  	  = NULL;     
  driver->plot_line   	  = NULL;      
  driver->plot_points 	  = NULL;    
  driver->free_driver 	  = NULL;    
  driver->set_window_size = NULL;
  return driver;
}
