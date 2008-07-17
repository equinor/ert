#ifndef __PLOT_H__
#define __PLOT_H__
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
 * As of now, plplot is installed in the directory /h/masar/plplot-5.9.0/Complilation.
 * For more information on how this was compiled, see the Readme file in the src/ directory.
 *
 * \subsection example1 Example 1: Plotting of the function: y = sinc(x).
 * \include simple.c

 * \subsection example2 Example 2: Plotting some well known functions.
 * \include plot_test.c
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

#include <util.h>
#include <list.h>
#include <list_node.h>
#include <plplot/plplot.h>
#include <plplot/plplotP.h>
#include <plplot/plplotcanvas.h>

/**
 * @brief Contains information about a plotting window.
 */
typedef struct plot_struct plot_type;


typedef enum plot_window_enum {
    NORMAL = 0,
    CANVAS = 1
} plot_window_type;

/**
 * @brief: Plot style for one single graph/dataset.
 * 
 * When adding a new dataset to your plot item you can define what plot style
 * you want for that one graph.
 */
typedef enum plot_style_enum {
    HISTOGRAM = 0,
    LINE = 1,
    POINT = 2
} plot_style_type;

/**
 * @brief: Plot color for one single graph/dataset.
 * 
 * When adding a new dataset to your plot item you can define what plot color
 * you want for that one graph. This color is also used to define color on labels
 * and titles.
 */
typedef enum plot_color_enum {
    WHITE = 0,
    RED = 1,
    YELLOW = 2,
    GREEN = 3,
    AQUAMARINE = 4,
    PINK = 5,
    WHEAT = 6,
    GRAY = 7,
    BROWN = 8,
    BLUE = 9,
    VIOLET = 10,
    CYAN = 11,
    TURQUOISE = 12,
    MAGENTA = 13,
    SALMON = 14,
    BLACK = 15
} plot_color_type;

extern plot_type *plot_alloc();
extern int plot_get_stream(plot_type * item);
extern PlplotCanvas *plot_get_canvas(plot_type * item);
extern list_type *plot_get_datasets(plot_type * item);
extern void plot_initialize(plot_type * item, const char *dev,
			    const char *filename, plot_window_type w);
extern void plot_set_labels(plot_type * item, const char *xlabel,
			    const char *ylabel, const char *title,
			    plot_color_type color);
extern void plot_set_viewport(plot_type * item, PLFLT xmin, PLFLT xmax,
			      PLFLT ymin, PLFLT ymax);
extern void plot_data(plot_type * item);
extern void plot_free_all_datasets(plot_type * item);
extern void plot_free(plot_type * item);

/**
 * @}
 */

#endif
