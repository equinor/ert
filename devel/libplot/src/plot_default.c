#include <util.h>
#include <plot_default.h>
#include <plot_const.h>

struct plot_dataset_default_struct {
  plot_style_type      style;        	  /**< The graph style: line|points|line_points */
  plot_color_type      line_color;   	  /**< The color for the line part of the plot. */
  plot_color_type      point_color;  	  /**< The color for the points in the plot. */
  plot_symbol_type     symbol_type;  	  /**< The type of symbol. */
  plot_line_style_type line_style;   	  /**< The style for lines. */

  
  double               symbol_size_base;  /**< These are the default values - when manipulating you use a scale factor. */
  double               line_width_base;   
  double               symbol_size_scale;
  double               lne_width_scale;
};

  


plot_dataset_default_type * plot_dataset_default_alloc() {
  plot_dataset_default_type * default = util_malloc( sizeof * default);
  default->style       = LINE;
  default->line_color  = BLUE;
  default->point_color = BLUE;
  default->symbol_type = 17;
  default->line_style  = solid_line;

  default->symbol_size_base  = 1.10;   
  default->line_width_base   = 1.5; 
  default->symbol_size_scale = 1.0;
  default->line_width_scale  = 1.0;
  return default;
}



void plot_dataset_default_set_style(plot_dataset_default_type * dataset , plot_style_type style) {
  dataset->style = style;
}


void plot_dataset_default_set_line_color(plot_dataset_default_type * dataset , plot_color_type line_color) {
  dataset->line_color = line_color;
}


void plot_dataset_default_set_point_color(plot_dataset_default_type * dataset , plot_color_type point_color) {
  dataset->point_color = point_color;
}


void plot_dataset_default_set_line_style(plot_dataset_default_type * dataset , plot_line_style_type line_style) {
  dataset->line_style = line_style;
}

void plot_dataset_default_set_symbol_size(plot_dataset_default_type * dataset , double symbol_size_scale) {
  dataset->symbol_size_scale = symbol_size_scale;
}


void plot_dataset_default_set_line_width(plot_dataset_default_type * dataset , double line_width_scale) {
  dataset->symbol_size_scale = line_width_scale;
}



