#ifndef __PLOT_DRIVER_H__
#define __PLOT_DRIVER_H__

#include <plot_dataset.h>
#include <plot_const.h>

typedef struct line_attribute_struct  line_attribute_type;
typedef struct point_attribute_struct point_attribute_type;
typedef struct plot_driver_struct     plot_driver_type;

typedef void (set_range_ftype)        (void * driver , double xmin , double xmax , double ymin , double ymax);
typedef void (set_label_ftype)        (void * driver , const char *);
typedef void (plot_line_ftype)        (void * driver , line_attribute_type  * , int , const double * , const double *);
typedef void (plot_point_ftype)       (void * driver , point_attribute_type * , int , const double * , const double *);
typedef void (window_size_ftype)      (void * driver , int width , int heigth);
typedef void (free_plot_driver_ftype) (void * driver );



struct plot_driver_struct {
  plot_driver_enum     driver_id; 
  void               	  * state;	       
  set_range_ftype    	  * set_range;      
  set_label_ftype    	  * set_title;      
  set_label_ftype    	  * set_xlabel;     
  set_label_ftype    	  * set_ylabel;     
  plot_line_ftype    	  * plot_line;      
  plot_point_ftype   	  * plot_points;    
  free_plot_driver_ftype  * free_driver;    
  window_size_ftype       * set_window_size;
};

struct line_attribute_struct {
  plot_color_type      line_color;   /**< The color for the line part of the plot. */
  plot_line_style_type line_style;   /**< The style for lines. */
  double               line_width;   /**< Scale factor for line width  : starts with PLOT_DEFAULT_LINE_WIDTH */
};





struct point_attribute_struct {
  plot_color_type      point_color;  /**< The color for the points in the plot. */
  plot_symbol_type     symbol_type;  /**< The type of symbol. */
  double               symbol_size;  /**< Scale factor for symbol size : starts with PLOT_DEFAULT_SYMBOL_SIZE */
};




plot_driver_type * plot_driver_alloc_empty(plot_driver_enum driver_id);

#endif
