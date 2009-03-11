#ifndef __PLOT_H__
#define __PLOT_H__
#ifdef __cplusplus
extern "C" {
#endif

/*! \mainpage libplot: plotting library based on Plplot
 *
 * \section intro Introduction
 *
 * This is a plotting library/wrapper for the plplot library. It is supposed to make life easier
 * when doing plots from the EnKF system.
 * Here is an example of what could be produced:
 * @image html punqs3_fopt.png 
 *
 * \section install Installation of plplot
 * As of now, plplot is installed in the directory /h/masar/plplot/.
 * For more information on how this was compiled, see the Readme file in the src/ directory.
 *
 * \subsection example1 Example 1: Plotting of the function: y = sinc(x).
 * \include simple.c

 * \subsection example2 Example 2: Plotting some well known functions.
 * \include plot_test.c
 * 
 * \subsection example3 Example 3: Plotting inside a gtk application, with animation.
 * \include plot_canvas_test.c
 */

/**
 * @addtogroup plot_type plot_type: A new unique plot with room for n datasets.
 * @brief Defines the plot_type, this is the core for every new plot you 
 * want to make. 
 *
 * @remarks For every new plot, start with plot_alloc() to create a new plot_type.
 *
 * @{
 */
#include <stdbool.h>
#include <util.h>
#include <plplot/plplot.h>
#include <plplot/plplotP.h>
#include <plot_const.h>
#include <plot_dataset.h>


typedef struct plot_struct plot_type;


plot_dataset_type * plot_alloc_new_dataset(plot_type *  , plot_data_type , bool);

plot_type *plot_alloc();
int plot_get_stream(plot_type * item);

 void plot_initialize(plot_type * item, const char *dev,
			    const char *filename);
 void plot_set_xlabel(plot_type * , const char *);
 void plot_set_ylabel(plot_type * , const char *);
 void plot_set_title(plot_type * , const char *);
 void plot_set_labels(plot_type * item, const char *xlabel, const char *ylabel, const char *title);

 void plot_data(plot_type * item);
 void plot_free(plot_type * item);
 void plot_get_extrema(plot_type * item, plot_range_type * );

  int plot_get_num_datasets(plot_type* item) ;
  plot_dataset_type** plot_get_datasets(plot_type* item) ;

void plot_set_window_size(plot_type * , int , int );
void plot_invert_y_axis(plot_type * );
void plot_invert_x_axis(plot_type * );

void plot_set_top_padding(plot_type    *  , double );
void plot_set_bottom_padding(plot_type *  , double );
void plot_set_left_padding(plot_type   *  , double );
void plot_set_right_padding(plot_type  *  , double );

void plot_set_label_color(plot_type * , plot_color_type );
void plot_set_box_color(plot_type *   , plot_color_type );
void plot_set_label_fontsize(plot_type * , double );

/**
 * @}
 */

#ifdef __cplusplus
}
#endif
#endif
